"""
03_lgbm_threshold_data.py — LightGBM-Vorhersagen auf dem 10.000er-Eval-Set

Bildet das Vorgehen aus ml_model_eval.py des Modellierungs-Teams exakt nach
(nur fuer LightGBM) und exportiert zwei kleine Dateien:

    results/lgbm_eval_predictions.parquet   amt, y_true, y_pred_proba
        -> Seite "Schwellwert und Kosten" (Threshold-Kurve, Kostenrechnung)

    results/lgbm_case_explorer.parquet      Anzeige-Merkmale + y_true + Proba
        -> Seite "Case Explorer" (Einzelfaelle auf dem 10.000er-Eval-Set)

Die Zahlen passen zur Modellvergleichsseite (Arbeitspunkt 0,65), sofern die
lokale lightgbm-Version der des Team-Laufs entspricht (siehe Verifikation).

Voraussetzungen:
  - datasets/train_engineered.parquet und datasets/eval_engineered.parquet
    im Modeling-Ordner (Pfad: config.MODELING_DIR)
  - pip install lightgbm scikit-learn

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
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix, precision_score, recall_score
except ImportError as e:
    sys.exit(f"Fehlendes Paket ({e.name}). Bitte installieren:  pip install lightgbm scikit-learn")

# Erwartete Werte aus ml_results.txt (Arbeitspunkt 0,65) zur Verifikation
EXPECTED = {"threshold": 0.65, "precision": 0.8298, "recall": 0.7091, "cm": [[9937, 8], [16, 39]]}


def find(path_candidates):
    for c in path_candidates:
        if c.exists():
            return c
    return None


def main() -> None:
    ds_dir = cfg.MODELING_DIR / "datasets"
    train_path = find([ds_dir / "train_engineered.parquet",
                       cfg.MODELING_DIR / "train_engineered.parquet"])
    eval_path = find([ds_dir / "eval_engineered.parquet",
                      cfg.MODELING_DIR / "eval_engineered.parquet"])
    if not train_path or not eval_path:
        sys.exit(
            "train_engineered.parquet / eval_engineered.parquet nicht gefunden.\n"
            f"Erwartet in: {ds_dir}\n"
            "Pfad anpassen in config.py (MODELING_DIR) oder Umgebungsvariable FRAUD_MODELING_DIR."
        )

    print(f"Lade TRAIN: {train_path}")
    df_train_raw = pd.read_parquet(train_path)
    print(f"Lade EVAL : {eval_path}")
    df_test_raw = pd.read_parquet(eval_path)
    print(f"  TRAIN {len(df_train_raw):,} Zeilen | EVAL {len(df_test_raw):,} Zeilen "
          f"({int(df_test_raw['is_fraud'].sum())} Betrug)")

    # Betraege des Eval-Sets fuer die spaetere Kostenrechnung sichern
    eval_amt = df_test_raw["amt"].to_numpy(copy=True)

    # Original-Anzeigespalten fuer den Case Explorer sichern (unkodiert, lesbar)
    display_cols = [c for c in ["amt", "hour", "category", "gender", "amt_ratio_7d",
                                "velocity_1h", "dist_km", "age", "city_pop_x",
                                "day_of_week"] if c in df_test_raw.columns]
    eval_display = df_test_raw[display_cols].reset_index(drop=True).copy()

    # ---- Ab hier exakt wie ml_model_eval.py -------------------------------
    df_train_raw = df_train_raw.copy()
    df_test_raw = df_test_raw.copy()
    if "row_id" in df_train_raw.columns:
        df_train_raw.pop("row_id")
    if "row_id" in df_test_raw.columns:
        df_test_raw.pop("row_id")

    df_all = pd.concat([df_train_raw, df_test_raw], axis=0, ignore_index=True)
    df_all = pd.get_dummies(
        df_all, columns=[c for c in ["category", "gender"] if c in df_all.columns],
        drop_first=True,
    )
    df_train = df_all.iloc[: len(df_train_raw)].copy()
    df_test = df_all.iloc[len(df_train_raw):].copy()

    X_train_full = df_train.drop(columns=["is_fraud"])
    y_train_full = df_train["is_fraud"].astype(int)
    X_test = df_test.drop(columns=["is_fraud"])
    y_test = df_test["is_fraud"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    params = {"class_weight": "balanced", "random_state": 42}
    tuned = cfg.MODELING_DIR / "debug" / "lgbm_best_params.json"
    if tuned.exists():
        params.update(json.loads(tuned.read_text()))
        print(f"  Getunte Parameter geladen: {tuned}")
    else:
        print("  Keine Tuning-Datei gefunden, Standard-Parameter (wie im Team-Lauf).")

    print("Trainiere LightGBM ...")
    model = LGBMClassifier(**params)
    model.fit(X_train_scaled, y_train)

    print("Berechne Vorhersage-Wahrscheinlichkeiten fuer das Eval-Set ...")
    proba = model.predict_proba(X_test_scaled)[:, 1]

    # ---- Verifikation gegen die geloggten Zahlen ---------------------------
    t = EXPECTED["threshold"]
    pred = (proba >= t).astype(int)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    cm = confusion_matrix(y_test, pred)
    print("\nVerifikation am Arbeitspunkt 0,65 (Soll aus ml_results.txt):")
    print(f"  Precision: {prec:.4f}  (Soll {EXPECTED['precision']:.4f})")
    print(f"  Recall   : {rec:.4f}  (Soll {EXPECTED['recall']:.4f})")
    print(f"  Matrix   : {cm.tolist()}  (Soll {EXPECTED['cm']})")
    if cm.tolist() != EXPECTED["cm"]:
        print("  [HINWEIS] Werte weichen vom Log ab. Moegliche Ursachen: andere "
              "lightgbm-Version als beim Team-Lauf oder vorhandene/fehlende "
              "Tuning-Datei. Fuer das Dashboard ist das unkritisch, die Seite "
              "bleibt in sich konsistent; fuer exakte Uebereinstimmung dieselbe "
              "Umgebung wie beim Team-Lauf verwenden.")
    else:
        print("  Exakte Uebereinstimmung mit dem Modellvergleich.")

    # ---- Export -------------------------------------------------------------
    out = pd.DataFrame({
        "amt": eval_amt.astype("float32"),
        "y_true": y_test.to_numpy().astype("int8"),
        "y_pred_proba": proba.astype("float32"),
    })
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cfg.LGBM_EVAL_PREDICTIONS_PATH, index=False)

    case = eval_display.copy()
    case["y_true"] = y_test.to_numpy().astype("int8")
    case["y_pred_proba"] = proba.astype("float32")
    case.to_parquet(cfg.CASE_EXPLORER_PATH, index=False)

    print("\n" + "=" * 64)
    print(f"Geschrieben: {cfg.LGBM_EVAL_PREDICTIONS_PATH}")
    print(f"  {len(out):,} Zeilen, {cfg.LGBM_EVAL_PREDICTIONS_PATH.stat().st_size/1024:.0f} KB")
    print(f"Geschrieben: {cfg.CASE_EXPLORER_PATH}")
    print(f"  {len(case):,} Zeilen, {cfg.CASE_EXPLORER_PATH.stat().st_size/1024:.0f} KB")
    print("  Beide Dateien sind klein und kommen mit ins Git-Repo (results/).")
    print("=" * 64)


if __name__ == "__main__":
    main()
