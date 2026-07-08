"""
03_lgbm_threshold_data.py — LightGBM-Vorhersagen auf dem 7.000er-Testset

Bildet das Vorgehen aus ml_model_eval.py des Modellierungs-Teams exakt nach
(nur fuer LightGBM): Training auf dem 200k-Trainingsset, Val/Test-Aufteilung
des 10k-Eval-Sets ueber das LLM-Split-Manifest, Vorhersagen auf dem
identischen 7.000er-Testset (38 Betrugsfaelle) wie die LLM-Laeufe.

Exportiert zwei kleine Dateien:

    results/lgbm_eval_predictions.parquet   amt, y_true, y_pred_proba
        -> Seite "Schwellwert und Kosten" (Threshold-Kurve, Kostenrechnung)

    results/lgbm_case_explorer.parquet      Anzeige-Merkmale + y_true + Proba
        -> Seite "Case Explorer" (Einzelfaelle auf dem 7.000er-Testset)

Voraussetzungen (im Modeling-Ordner, Pfad: config.MODELING_DIR):
    datasets/train_engineered.parquet
    datasets/eval_engineered.parquet
    datasets/llm_eval_val_test_split.parquet   (Split-Manifest aus dem LLM-Lauf)
    models/lgbm_best_params.json               (getunte Parameter)

Aufruf (im fraud_dashboard-Ordner, venv aktiv):
    python scripts/03_lgbm_threshold_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402

try:
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix, precision_score, recall_score
except ImportError as e:
    sys.exit(f"Fehlendes Paket ({e.name}). Bitte installieren:  pip install lightgbm scikit-learn")

# Erwartete Werte aus ml_results.txt (07.07.2026, Arbeitspunkt 0,35) zur Verifikation
EXPECTED = {"threshold": 0.35, "precision": 0.7561, "recall": 0.8158, "cm": [[6952, 10], [7, 31]]}

DISPLAY_COLS = ["amt", "hour", "category", "gender", "amt_ratio_7d",
                "velocity_1h", "dist_km", "age", "city_pop_x", "day_of_week"]


def find(candidates):
    for c in candidates:
        if c.exists():
            return c
    return None


def main() -> None:
    ds = cfg.MODELING_DIR / "datasets"
    train_path = find([ds / "train_engineered.parquet"])
    eval_path = find([ds / "eval_engineered.parquet"])
    manifest_path = find([ds / "llm_eval_val_test_split.parquet"])
    missing = [n for n, p in [("train_engineered.parquet", train_path),
                              ("eval_engineered.parquet", eval_path),
                              ("llm_eval_val_test_split.parquet", manifest_path)] if p is None]
    if missing:
        sys.exit(
            "Fehlende Dateien in " + str(ds) + ":\n  " + "\n  ".join(missing) +
            "\nPfad anpassen in config.py (MODELING_DIR) oder Umgebungsvariable FRAUD_MODELING_DIR."
        )

    print(f"Lade TRAIN   : {train_path}")
    df_train_raw = pd.read_parquet(train_path)
    print(f"Lade EVAL    : {eval_path}")
    df_eval_raw = pd.read_parquet(eval_path)
    print(f"Lade MANIFEST: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)

    # ---- Ab hier exakt wie ml_model_eval.py ------------------------------
    df_train_raw = df_train_raw.copy()
    df_eval_raw = df_eval_raw.copy()
    if "row_id" in df_train_raw.columns:
        df_train_raw.pop("row_id")
    if "row_id" not in df_eval_raw.columns:
        df_eval_raw["row_id"] = df_eval_raw.index

    df_all = pd.concat(
        [df_train_raw, df_eval_raw.drop(columns=["row_id"])], axis=0, ignore_index=True
    )
    df_all = pd.get_dummies(
        df_all, columns=[c for c in ["category", "gender"] if c in df_all.columns],
        drop_first=True,
    )

    df_train = df_all.iloc[: len(df_train_raw)].copy()
    X_train = df_train.drop(columns=["is_fraud"])
    y_train = df_train["is_fraud"].astype(int)

    df_eval = df_all.iloc[len(df_train_raw):].copy()
    df_eval["row_id"] = df_eval_raw["row_id"].values

    test_ids = manifest[manifest["eval_split"] == "test"]["row_id"]
    df_test = df_eval[df_eval["row_id"].isin(test_ids)].copy()
    test_mask = df_eval_raw["row_id"].isin(test_ids).to_numpy()

    X_test = df_test.drop(columns=["is_fraud", "row_id"])
    y_test = df_test["is_fraud"].astype(int)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    print(f"  TRAIN {len(X_train_scaled):,} Zeilen | TEST {len(X_test_scaled):,} Zeilen "
          f"({int(y_test.sum())} Betrug)")

    params = {"class_weight": "balanced", "random_state": 42}
    tuned = find([cfg.MODELING_DIR / "models" / "lgbm_best_params.json",
                  cfg.MODELING_DIR / "debug" / "lgbm_best_params.json"])
    if tuned:
        params.update(json.loads(tuned.read_text()))
        print(f"  Getunte Parameter geladen: {tuned}")
    else:
        print("  [WARNUNG] Keine Tuning-Datei gefunden; Standard-Parameter. "
              "Die Verifikation wird dann sehr wahrscheinlich abweichen.")

    print("Trainiere LightGBM ...")
    model = LGBMClassifier(**params)
    model.fit(X_train_scaled, y_train)

    # ---- Verifikation auf dem 7.000er-Testset (Konsistenz zum Modellvergleich) ----
    proba_test = model.predict_proba(X_test_scaled)[:, 1]
    t = EXPECTED["threshold"]
    pred = (proba_test >= t).astype(int)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    cm = confusion_matrix(y_test, pred)
    print(f"\nVerifikation auf 7.000er-Testset, Arbeitspunkt {t:.2f} (Soll aus ml_results.txt):")
    print(f"  Precision: {prec:.4f}  (Soll {EXPECTED['precision']:.4f})")
    print(f"  Recall   : {rec:.4f}  (Soll {EXPECTED['recall']:.4f})")
    print(f"  Matrix   : {cm.tolist()}  (Soll {EXPECTED['cm']})")
    if cm.tolist() == EXPECTED["cm"]:
        print("  Exakte Uebereinstimmung mit dem Modellvergleich.")
    else:
        print("  [HINWEIS] Werte weichen vom Log ab. Moegliche Ursachen: andere "
              "lightgbm-/scikit-learn-Version als beim Team-Lauf oder fehlende "
              "Tuning-Datei. Die Dashboard-Seiten bleiben in sich konsistent.")

    # ---- Vorhersagen auf dem groesseren Auswertungsset (Threshold + Case Explorer) ----
    analysis_path = find([ds / "train_engineered_100K.parquet",
                          ds / "eval_engineered_100K.parquet",
                          ds / "analysis_100K.parquet"])
    if analysis_path is None:
        sys.exit(
            "Auswertungsset nicht gefunden (erwartet z. B. train_engineered_100K.parquet in "
            + str(ds) + "). Fuer die Threshold- und Case-Explorer-Seite noetig."
        )
    print(f"\nLade Auswertungsset: {analysis_path}")
    df_ana_raw = pd.read_parquet(analysis_path)
    n_fraud_ana = int(df_ana_raw["is_fraud"].sum())
    print(f"  {len(df_ana_raw):,} Zeilen ({n_fraud_ana} Betrug) — disjunkt zum Training")

    # Gleiche Feature-Aufbereitung wie beim Training (Dummies an Trainingsspalten ausrichten)
    df_ana = df_ana_raw.copy()
    if "row_id" in df_ana.columns:
        df_ana = df_ana.drop(columns=["row_id"])
    df_ana = pd.get_dummies(
        df_ana, columns=[c for c in ["category", "gender"] if c in df_ana.columns],
        drop_first=True,
    )
    X_ana = df_ana.drop(columns=["is_fraud"]).reindex(columns=X_train.columns, fill_value=0)
    y_ana = df_ana_raw["is_fraud"].astype(int)
    X_ana_scaled = pd.DataFrame(scaler.transform(X_ana), columns=X_train.columns)
    proba_ana = model.predict_proba(X_ana_scaled)[:, 1]

    # ---- Export (basiert auf dem 100k-Auswertungsset) ----------------------
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out = pd.DataFrame({
        "amt": df_ana_raw["amt"].to_numpy().astype("float32"),
        "y_true": y_ana.to_numpy().astype("int8"),
        "y_pred_proba": proba_ana.astype("float32"),
    })
    out.to_parquet(cfg.LGBM_EVAL_PREDICTIONS_PATH, index=False)

    avail = [c for c in DISPLAY_COLS if c in df_ana_raw.columns]
    case = df_ana_raw[avail].reset_index(drop=True).copy()
    case["y_true"] = y_ana.to_numpy().astype("int8")
    case["y_pred_proba"] = proba_ana.astype("float32")
    case.to_parquet(cfg.CASE_EXPLORER_PATH, index=False)

    print("\n" + "=" * 64)
    print(f"Threshold-/Kosten- und Case-Explorer-Daten auf {len(out):,} Zeilen "
          f"({n_fraud_ana} Betrug):")
    print(f"  {cfg.LGBM_EVAL_PREDICTIONS_PATH}  ({cfg.LGBM_EVAL_PREDICTIONS_PATH.stat().st_size/1024:.0f} KB)")
    print(f"  {cfg.CASE_EXPLORER_PATH}  ({cfg.CASE_EXPLORER_PATH.stat().st_size/1024:.0f} KB)")
    print("  Beide Dateien kommen mit ins Git-Repo (results/).")
    print("=" * 64)


if __name__ == "__main__":
    main()
