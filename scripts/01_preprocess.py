"""
01_preprocess.py — Offline-Pipeline: Sparkov-Rohdaten -> kompakte Analyse-Artefakte

Liest alle Transaktions-CSVs aus RAW_DATA_DIR chunked ein und erzeugt:

  data/processed/transactions/part_XXX.parquet   schlanke, maskierte Transaktionen
                                                 inkl. abgeleiteter Features
  data/processed/aggregates/*.parquet            voraggregierte Tabellen für die
                                                 EDA-Charts der App (winzig)
  data/processed/plot_sample.parquet             stratifiziertes Sample für
                                                 Verteilungs-Plots (Fraud komplett
                                                 bzw. gedeckelt + 1% Non-Fraud)
  data/processed/card_stats.parquet              Betrags-Statistik je Karte
                                                 (Basis für z-Score-Features)
  data/processed/meta.json                       Kennzahlen (Fraud-Rate, Zeitraum, ...)

Aufruf:   python scripts/01_preprocess.py
Dauer:    bei ~5 GB CSV grob 5–15 Minuten, RAM-Bedarf gering (chunked).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Projektwurzel importierbar machen, egal von wo das Skript gestartet wird
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def is_transaction_file(path: Path) -> bool:
    """Transaktionsdateien erkennen wir am Spaltennamen 'trans_num' im Header.
    So werden customers.csv, merchants.csv, demographics.csv etc. übersprungen."""
    try:
        header = path.open("r", encoding="utf-8", errors="replace").readline()
    except OSError:
        return False
    return "trans_num" in header and cfg.SEP in header


def mask_card(cc_num: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Kreditkartennummer -> (stabile anonyme ID, Anzeige-Label '•••• 1234')."""
    s = cc_num.astype(str)
    card_id = s.map(lambda x: hashlib.sha256(x.encode()).hexdigest()[:12])
    card_label = "•••• " + s.str[-4:]
    return card_id, card_label


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distanz Kunde<->Händler in km (vektorisiert)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def parse_profile(profile: pd.Series) -> pd.DataFrame:
    """'adults_50up_male_rural.json' -> Altersgruppe + Wohngegend (lesbar)."""
    stem = profile.str.replace(".json", "", regex=False)
    area = stem.str.rsplit("_", n=2).str[-1].map(cfg.AREA_LABELS)
    age_key = stem.str.rsplit("_", n=2).str[0]  # z. B. 'adults_50up'
    age_group = age_key.map(cfg.AGE_GROUP_LABELS)
    return pd.DataFrame({"age_group": age_group, "area": area, "segment": stem})


def transform_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Rohdaten-Chunk -> maskierter, angereicherter Chunk."""
    out = pd.DataFrame()

    # --- Identität (maskiert) ---
    out["card_id"], out["card_label"] = mask_card(chunk["cc_num"])
    out["gender"] = chunk["gender"]
    out["state"] = chunk["state"]
    out["city_pop"] = chunk["city_pop"].astype("int32")
    out["job"] = chunk["job"]

    # --- Segment aus Sparkov-Profil ---
    out = pd.concat([out, parse_profile(chunk["profile"])], axis=1)

    # --- Zeit-Features ---
    ts = pd.to_datetime(
        chunk["trans_date"] + " " + chunk["trans_time"], format="%Y-%m-%d %H:%M:%S"
    )
    out["ts"] = ts
    out["date"] = ts.dt.date.astype("str")
    out["month"] = ts.dt.to_period("M").astype(str)
    out["hour"] = ts.dt.hour.astype("int8")
    out["weekday"] = ts.dt.dayofweek.map(dict(enumerate(cfg.WEEKDAYS_DE)))

    # --- Alter zum Transaktionszeitpunkt ---
    dob = pd.to_datetime(chunk["dob"], format="%Y-%m-%d")
    out["age"] = ((ts - dob).dt.days / 365.25).astype("int16")

    # --- Transaktion ---
    out["trans_num"] = chunk["trans_num"]
    out["category"] = chunk["category"]
    out["amt"] = chunk["amt"].astype("float32")
    # Faker-Artefakt 'fraud_' im Händlernamen entfernen (hat NICHTS mit dem Label zu tun!)
    out["merchant"] = chunk["merchant"].str.removeprefix("fraud_")
    out["is_fraud"] = chunk["is_fraud"].astype("int8")

    # --- Geo ---
    out["cust_lat"] = chunk["lat"].astype("float32")
    out["cust_long"] = chunk["long"].astype("float32")
    out["merch_lat"] = chunk["merch_lat"].astype("float32")
    out["merch_long"] = chunk["merch_long"].astype("float32")
    out["distance_km"] = haversine_km(
        chunk["lat"], chunk["long"], chunk["merch_lat"], chunk["merch_long"]
    ).astype("float32")

    return out


# ---------------------------------------------------------------------------
# Aggregat-Akkumulatoren (werden chunkweise aufaddiert)
# ---------------------------------------------------------------------------

AGG_SPECS = {
    # Name -> Gruppierspalten
    "by_hour": ["hour"],
    "by_weekday": ["weekday"],
    "by_category": ["category"],
    "by_month": ["month"],
    "by_state": ["state"],
    "by_segment": ["age_group", "gender", "area"],
    "by_hour_category": ["hour", "category"],
}


def update_aggregates(store: dict, chunk: pd.DataFrame) -> None:
    for name, keys in AGG_SPECS.items():
        g = chunk.groupby(keys, observed=True).agg(
            n=("is_fraud", "size"),
            n_fraud=("is_fraud", "sum"),
            amt_sum=("amt", "sum"),
            fraud_amt_sum=("amt", lambda s: 0.0),  # placeholder, unten korrekt
        )
        # Fraud-Schadenssumme korrekt berechnen
        fr = chunk[chunk["is_fraud"] == 1].groupby(keys, observed=True)["amt"].sum()
        g["fraud_amt_sum"] = fr.reindex(g.index).fillna(0.0)
        store[name] = g if name not in store else store[name].add(g, fill_value=0)


def update_card_stats(store: dict, chunk: pd.DataFrame) -> None:
    g = chunk.groupby("card_id").agg(
        n=("amt", "size"),
        amt_sum=("amt", "sum"),
        amt_sq_sum=("amt", lambda s: float((s.astype("float64") ** 2).sum())),
        n_fraud=("is_fraud", "sum"),
    )
    store["cards"] = g if "cards" not in store else store["cards"].add(g, fill_value=0)


# ---------------------------------------------------------------------------
# Hauptlauf
# ---------------------------------------------------------------------------

def main() -> None:
    raw_dir = cfg.RAW_DATA_DIR
    if not raw_dir.exists():
        sys.exit(
            f"Rohdaten-Ordner nicht gefunden: {raw_dir}\n"
            "Pfad in config.py anpassen oder FRAUD_DATA_DIR setzen."
        )

    files = sorted(p for p in raw_dir.glob("*.csv") if is_transaction_file(p))
    if not files:
        sys.exit(f"Keine Transaktionsdateien (mit Spalte 'trans_num') in {raw_dir} gefunden.")
    print(f"{len(files)} Transaktionsdateien gefunden in {raw_dir}\n")

    cfg.TRANSACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.AGG_DIR.mkdir(parents=True, exist_ok=True)

    aggs: dict = {}
    cards: dict = {}
    fraud_samples: list[pd.DataFrame] = []
    legit_samples: list[pd.DataFrame] = []
    rng = np.random.default_rng(42)

    total_rows = 0
    total_fraud = 0
    ts_min, ts_max = None, None
    t0 = time.time()

    for i, path in enumerate(files):
        parts = []
        for chunk in pd.read_csv(
            path,
            sep=cfg.SEP,
            chunksize=cfg.CHUNKSIZE,
            dtype={"cc_num": str, "zip": str, "acct_num": str, "ssn": str},
        ):
            t = transform_chunk(chunk)
            parts.append(t)

            total_rows += len(t)
            total_fraud += int(t["is_fraud"].sum())
            lo, hi = t["ts"].min(), t["ts"].max()
            ts_min = lo if ts_min is None else min(ts_min, lo)
            ts_max = hi if ts_max is None else max(ts_max, hi)

            update_aggregates(aggs, t)
            update_card_stats(cards, t)

            fraud_samples.append(t[t["is_fraud"] == 1])
            legit = t[t["is_fraud"] == 0]
            take = legit.sample(frac=cfg.NONFRAUD_SAMPLE_FRAC, random_state=int(rng.integers(1e9)))
            legit_samples.append(take)

        out_path = cfg.TRANSACTIONS_DIR / f"part_{i:03d}.parquet"
        pd.concat(parts, ignore_index=True).to_parquet(out_path, index=False)
        print(
            f"[{i + 1:>2}/{len(files)}] {path.name:<55} -> {out_path.name}"
            f"  ({total_rows:,} Zeilen kumuliert, {time.time() - t0:,.0f}s)"
        )

    # --- Aggregate speichern ---
    for name, df in aggs.items():
        df = df.reset_index()
        df["fraud_rate"] = df["n_fraud"] / df["n"]
        df.to_parquet(cfg.AGG_DIR / f"{name}.parquet", index=False)

    # --- Karten-Statistik (Basis für z-Score-Feature der Modellierung) ---
    c = cards["cards"]
    c["amt_mean"] = c["amt_sum"] / c["n"]
    var = c["amt_sq_sum"] / c["n"] - c["amt_mean"] ** 2
    c["amt_std"] = np.sqrt(var.clip(lower=0))
    c.reset_index().to_parquet(cfg.CARD_STATS_PATH, index=False)

    # --- Plot-Sample (für Verteilungs-Plots in App & Notebook) ---
    fraud_df = pd.concat(fraud_samples, ignore_index=True)
    legit_df = pd.concat(legit_samples, ignore_index=True)
    if len(fraud_df) > cfg.PLOT_SAMPLE_MAX_ROWS:
        fraud_df = fraud_df.sample(cfg.PLOT_SAMPLE_MAX_ROWS, random_state=42)
    if len(legit_df) > cfg.PLOT_SAMPLE_MAX_ROWS:
        legit_df = legit_df.sample(cfg.PLOT_SAMPLE_MAX_ROWS, random_state=42)
    sample = pd.concat([fraud_df, legit_df], ignore_index=True)
    sample.to_parquet(cfg.PLOT_SAMPLE_PATH, index=False)

    # --- Meta-Kennzahlen ---
    meta = {
        "n_transactions": int(total_rows),
        "n_fraud": int(total_fraud),
        "fraud_rate": total_fraud / total_rows,
        "n_cards": int(len(c)),
        "ts_min": str(ts_min),
        "ts_max": str(ts_max),
        "n_source_files": len(files),
        "plot_sample_rows": int(len(sample)),
        "nonfraud_sample_frac": cfg.NONFRAUD_SAMPLE_FRAC,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cfg.META_PATH.write_text(json.dumps(meta, indent=2))

    print("\n" + "=" * 70)
    print(f"Fertig in {time.time() - t0:,.0f}s")
    print(f"  Transaktionen : {total_rows:>12,}")
    print(f"  davon Fraud   : {total_fraud:>12,}  ({total_fraud / total_rows:.3%})")
    print(f"  Karten        : {len(c):>12,}")
    print(f"  Zeitraum      : {ts_min}  bis  {ts_max}")
    print(f"  Output        : {cfg.PROCESSED_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
