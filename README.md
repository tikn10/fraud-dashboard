# Kreditkarten-Fraud-Detection — Dashboard & Daten-Pipeline

Uni-Projekt: Analyse und Präsentation synthetischer Kreditkartentransaktionen
(Sparkov-Generator, 10k Kunden, 2 Jahre, ~5 GB Rohdaten) mit Streamlit.

## Projektstruktur

```
fraud_dashboard/
├── config.py                  # Pfade, Konstanten, Farbschema (zentral!)
├── requirements.txt
├── results/                   # Modell-Ergebnisse aus den Team-Logs (klein, im Repo)
│   ├── model_results.json
│   └── rf_threshold_curve.csv
├── scripts/
│   ├── 01_preprocess.py       # Pipeline: 60 Roh-CSVs -> Parquet + Aggregate
│   └── 02_train_and_predict.py# RF nachtrainieren -> Fallbeispiele für Case Explorer
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

## 2b) Fallbeispiele für den Case Explorer erzeugen (optional, einmalig)

Der Case Explorer braucht Vorhersagen pro Transaktion. Diese erzeugt ein Skript,
das den Random Forest auf dem Modeling-Parquet des Teams nachtrainiert:

```bat
pip install scikit-learn
python scripts\02_train_and_predict.py
```

Pfad zum Modeling-Parquet in `config.py` -> `MODELING_DIR` (Standard:
`...\credit_card_fraud\modeling`). Das Skript schreibt
`data/processed/case_explorer.parquet` (klein, kommt mit ins Repo). Ohne diese
Datei zeigt die Case-Explorer-Seite eine Anleitung statt eines Fehlers.

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
