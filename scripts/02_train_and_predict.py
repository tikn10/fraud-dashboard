"""
02_train_and_predict.py — erzeugt die Fallbeispiel-Datei für den Case Explorer

Trainiert das beste Modell (Random Forest) auf dem Modeling-Sample des Teams
nach, berechnet die Vorhersage-Wahrscheinlichkeit pro Test-Transaktion und
schreibt eine kompakte Datei fürs Dashboard:

    data/processed/case_explorer.parquet

Diese Datei ist klein (nur ein stratifiziertes Sample des Testsets) und kann
mit ins GitHub-Repo — das 606-MB-Modeling-Parquet bleibt lokal.

Voraussetzung: Das Modeling-Parquet des Teams liegt vor. Standardmäßig wird
zuerst nach dem 5%-Sample gesucht (schnell, deckt sich mit den geloggten
Metriken), sonst nach dem Vollfile.

Aufruf (im venv, im fraud_dashboard-Ordner):
    python scripts/02_train_and_predict.py

Benötigt scikit-learn:  pip install scikit-learn
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
except ImportError:
    sys.exit("scikit-learn fehlt. Bitte installieren:  pip install scikit-learn")

# Spalten, die fürs Anzeigen im Case Explorer übernommen werden (sofern vorhanden)
DISPLAY_COLS = [
    "amt", "hour", "category", "gender", "amt_ratio_7d",
    "velocity_1h", "dist_km", "age", "city_pop_x", "day_of_week",
]
SAMPLE_TARGET_ROWS = 15_000   # Zielgröße der Ausgabedatei (alle Frauds + Sample Non-Fraud)


def find_modeling_parquet() -> Path:
    """Sucht das Modeling-Parquet: erst SAMPLE (schnell), dann FULL."""
    candidates = [
        cfg.MODELING_DIR / "debug" / "filtered_fraud_data_SAMPLE.parquet",
        cfg.MODELING_DIR / "filtered_fraud_data_SAMPLE.parquet",
        cfg.MODELING_DIR / "debug" / "filtered_fraud_data_FULL.parquet",
        cfg.MODELING_DIR / "filtered_fraud_data_FULL.parquet",
    ]
    for c in candidates:
        if c.exists():
            return c
    sys.exit(
        "Kein Modeling-Parquet gefunden. Erwartete Orte:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + f"\n\nPfad anpassen in config.py (MODELING_DIR) oder per Umgebungsvariable "
        f"FRAUD_MODELING_DIR. Aktuell: {cfg.MODELING_DIR}"
    )


def main() -> None:
    src = find_modeling_parquet()
    print(f"Lade Modeling-Daten: {src}")
    df = pd.read_parquet(src)
    print(f"  {len(df):,} Zeilen, {df['is_fraud'].sum():,} Fraud "
          f"({df['is_fraud'].mean():.3%})")

    # --- Modell-Matrix wie im Eval-Skript des Teams (get_dummies + object-Spalten raus) ---
    df_model = pd.get_dummies(
        df, columns=[c for c in ["category", "gender"] if c in df.columns], drop_first=True
    )
    obj_cols = df_model.select_dtypes(include="object").columns.tolist()
    df_model = df_model.drop(columns=obj_cols)

    X = df_model.drop(columns=["is_fraud"])
    y = df_model["is_fraud"]

    # Gleicher Split wie im Team-Skript (über den Index, stratify=y, seed 42)
    idx_train, idx_test = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Trainings-Set: {len(idx_train):,} | Test-Set: {len(idx_test):,}")

    # RF mit den geloggten Parametern (n_estimators=100, class_weight balanced)
    print("Trainiere Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X.loc[idx_train], y.loc[idx_train])

    print("Berechne Vorhersage-Wahrscheinlichkeiten fürs Test-Set …")
    proba = rf.predict_proba(X.loc[idx_test])[:, 1]

    # --- Anzeige-Frame aus den ORIGINAL-Werten (unskaliert, lesbar) ---
    avail = [c for c in DISPLAY_COLS if c in df.columns]
    out = df.loc[idx_test, avail].copy()
    out["y_true"] = y.loc[idx_test].to_numpy()
    out["y_pred_proba"] = proba

    # --- Stratifiziert verkleinern: alle Frauds behalten + Non-Fraud-Sample ---
    fraud = out[out["y_true"] == 1]
    legit = out[out["y_true"] == 0]
    n_legit = max(0, SAMPLE_TARGET_ROWS - len(fraud))
    if len(legit) > n_legit:
        legit = legit.sample(n_legit, random_state=42)
    final = pd.concat([fraud, legit]).sample(frac=1, random_state=42).reset_index(drop=True)

    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    final.to_parquet(cfg.CASE_EXPLORER_PATH, index=False)

    print("\n" + "=" * 60)
    print(f"Geschrieben: {cfg.CASE_EXPLORER_PATH}")
    print(f"  {len(final):,} Zeilen  ({final['y_true'].sum():,} Fraud)")
    print(f"  Spalten: {list(final.columns)}")
    print("Diese Datei kann mit ins GitHub-Repo (klein genug).")
    print("=" * 60)


if __name__ == "__main__":
    main()
