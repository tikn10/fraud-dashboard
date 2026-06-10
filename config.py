"""
Zentrale Konfiguration für das Fraud-Detection-Projekt.

Alle Pfade und Konstanten an einer Stelle, damit Skript, Notebook und
Streamlit-App dieselben Annahmen teilen.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
# Projektwurzel = Ordner, in dem diese Datei liegt
PROJECT_ROOT = Path(__file__).resolve().parent

# Rohdaten: Ordner mit den 60 Transaktions-CSVs + customers.csv.
# Kann über die Umgebungsvariable FRAUD_DATA_DIR überschrieben werden.
RAW_DATA_DIR = Path(
    os.environ.get(
        "FRAUD_DATA_DIR",
        r"C:\Users\Timon Knief\Desktop\credit_card_fraud\Sparkov_Data_Generation\generated_data_10k",
    )
)

# Aufbereitete Daten (Output der Pipeline, Input der App)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRANSACTIONS_DIR = PROCESSED_DIR / "transactions"   # Parquet-Parts
AGG_DIR = PROCESSED_DIR / "aggregates"              # kleine Aggregat-Tabellen
PLOT_SAMPLE_PATH = PROCESSED_DIR / "plot_sample.parquet"
CARD_STATS_PATH = PROCESSED_DIR / "card_stats.parquet"
META_PATH = PROCESSED_DIR / "meta.json"

# ---------------------------------------------------------------------------
# Daten-Konstanten (Sparkov-spezifisch)
# ---------------------------------------------------------------------------
SEP = "|"  # Sparkov schreibt pipe-separierte CSVs

# Spalten, die personenbezogen wirken und NICHT in die aufbereiteten
# Daten übernommen werden (Privacy by Design, auch wenn alles synthetisch ist)
PII_COLUMNS = ["ssn", "first", "last", "street", "zip", "acct_num", "cc_num"]

# Sparkov-Profile -> lesbare Segment-Labels
AGE_GROUP_LABELS = {
    "young_adults": "unter 25",
    "adults_2550": "25–50",
    "adults_50up": "über 50",
}
AREA_LABELS = {"urban": "urban", "rural": "ländlich"}

# Deutsche Wochentagsnamen in fester Reihenfolge (Mo–So)
WEEKDAYS_DE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

# ---------------------------------------------------------------------------
# Pipeline-Parameter
# ---------------------------------------------------------------------------
CHUNKSIZE = 250_000          # Zeilen pro Chunk beim CSV-Einlesen
NONFRAUD_SAMPLE_FRAC = 0.01  # Anteil legitimer Transaktionen im Plot-Sample
PLOT_SAMPLE_MAX_ROWS = 300_000  # Obergrenze je Klasse im Plot-Sample

# ---------------------------------------------------------------------------
# Farbschema (eine Signalfarbe für Fraud, eine neutrale für legitim –
# konsequent in App UND Notebook verwenden)
# ---------------------------------------------------------------------------
COLOR_FRAUD = "#F25C54"      # Signal: Korallrot
COLOR_LEGIT = "#5B8DB8"      # Neutral: Stahlblau
COLOR_ACCENT = "#E8A33D"     # Akzent: Bernstein (Interaktion/Hervorhebung)
COLOR_BG = "#0E1525"         # Hintergrund: tiefes Nachtblau
COLOR_BG2 = "#1A2332"        # Sekundärflächen
COLOR_TEXT = "#E8EDF5"
COLOR_GRID = "#2A3650"
