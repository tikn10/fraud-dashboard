# Kreditkarten-Fraud-Detection — Dashboard & Daten-Pipeline

Uni-Projekt: Analyse und Präsentation synthetischer Kreditkartentransaktionen
(Sparkov-Generator, 10k Kunden, 2 Jahre, ~5 GB Rohdaten) mit Streamlit.

## Projektstruktur

```
fraud_dashboard/
├── config.py                  # Pfade, Konstanten, Farbschema (zentral!)
├── requirements.txt
├── results/                   # Modell-Ergebnisse aus den Team-Logs (klein, im Repo)
│   ├── model_results.json     # finale ML-Metriken (4 Modelle, 10k-EVAL-Set)
│   ├── lgbm_eval_predictions.parquet  # LightGBM-Vorhersagen (aus Skript 03)
│   ├── lgbm_case_explorer.parquet     # Eval-Zeilen + Vorhersagen (aus Skript 03)
├── scripts/
│   ├── 01_preprocess.py       # Pipeline: 60 Roh-CSVs -> Parquet + Aggregate
│   └── 03_lgbm_threshold_data.py # LightGBM auf 7k-Testset -> Threshold-Seite + Case Explorer
├── notebooks/
│   └── 01_rohdaten_check.ipynb
├── app/
│   ├── Home.py                # Streamlit-Startseite
│   ├── utils.py               # Loader, Plotly-Theme, Helfer
│   ├── pages/
│   │   ├── 1_🗂️_Datensatz.py
│   │   ├── 2_🔍_Explorative_Analyse.py
│   │   ├── 3_🧬_Feature_Engineering.py
│   │   ├── 4_🏆_Modellvergleich.py
│   │   ├── 5_🎚️_Threshold_und_Kosten.py
│   │   └── 6_🔬_Case_Explorer.py
│   └── .streamlit/config.toml # Dunkles Theme
└── data/processed/            # Output der Pipeline (wird erzeugt)
```

## Setup (einmalig)

```bat
cd fraud_dashboard
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Rohdaten-Pfad prüfen: `config.py` -> `RAW_DATA_DIR`
(voreingestellt auf `...\Sparkov_Data_Generation\generated_data_10k`).

## 1) Pipeline ausführen (einmalig, ~5–15 min)

```bat
python scripts\01_preprocess.py
```

Erzeugt unter `data/processed/`:

| Artefakt | Inhalt | Zweck |
|---|---|---|
| `transactions/part_*.parquet` | alle Transaktionen, maskiert + Features | Modellierung, Case Explorer |
| `aggregates/*.parquet` | Fraud-Rate je Stunde/Kategorie/Segment/Monat/... | EDA-Charts (App lädt nur diese) |
| `plot_sample.parquet` | Fraud (gedeckelt) + 1 % Non-Fraud | Verteilungs-Plots |
| `card_stats.parquet` | Betragsstatistik je Karte | z-Score-Features (Modellierung) |
| `meta.json` | Kennzahlen (Fraud-Rate, Zeitraum, ...) | Kopfzahlen der App |

**Maskierung:** SSN, Name, Adresse, Konto- und Kartennummer werden entfernt;
Karten erscheinen nur als Hash-ID + letzte 4 Ziffern. Das Faker-Artefakt
`fraud_` im Händlernamen (hat nichts mit dem Label zu tun!) wird entfernt.

## 2) App starten

```bat
streamlit run app\Home.py
```

## 2b) Daten für Threshold-Seite und Case Explorer erzeugen (einmalig)

Die Seiten "Schwellwert und Kosten" und "Case Explorer" arbeiten mit
LightGBM-Vorhersagen auf dem größeren, im Training ungenutzten Auswertungsset.
Diese erzeugt (wie im Team-Lauf nachgebildet):

```bat
pip install lightgbm scikit-learn
python scripts\03_lgbm_threshold_data.py
```

Voraussetzung: `train_engineered.parquet`, `eval_engineered.parquet` und das
Split-Manifest `llm_eval_val_test_split.parquet` liegen im Modeling-Ordner unter
`datasets\`; die getunten Parameter (`lgbm_best_params.json`) unter `models\`
(Pfad in `config.py` -> `MODELING_DIR`).
Das Skript verifiziert die Zahlen gegen den geloggten Arbeitspunkt (0,35) und
schreibt `results/lgbm_eval_predictions.parquet` sowie
`results/lgbm_case_explorer.parquet` (beide klein, kommen mit ins Repo).
Meldet die Verifikation eine Abweichung, liegt das meist an einer anderen
lightgbm-Version als im Team-Lauf; für exakte Übereinstimmung dieselbe Version
installieren (`pip install lightgbm==<Version des Teams>`) und erneut ausführen.

## 3) Notebook (optional)

```bat
jupyter notebook notebooks\01_rohdaten_check.ipynb
```

## Übergabe an die Modellierung

Die Kolleg:innen lesen einfach `data/processed/transactions/` als ein
DataFrame (`pd.read_parquet(...)`) — Features wie `hour`, `weekday`, `age`,
`age_group`, `area`, `distance_km` sind bereits enthalten; `card_stats.parquet`
liefert die Basis für kartenrelative Betrags-Features.

**Gewünschtes Format der Modell-Outputs** (pro Modell eine Datei):
`trans_num, y_true, y_pred_proba` — daraus baut die App später
Modellvergleich, Threshold-Slider und Case Explorer.

Empfehlung: **zeitbasierter** Train/Test-Split (Jahr 1 train, Jahr 2 test)
statt zufällig, um Leakage zu vermeiden und den Realeinsatz zu simulieren.
