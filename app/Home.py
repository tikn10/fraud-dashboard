"""Startseite: Projekt-Framing + Kernzahlen. Aufruf:  streamlit run app/Home.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as u  # noqa: E402

u.page_setup("Übersicht")

st.title("🛡️ Kreditkarten-Betrugserkennung")
st.caption(
    "Uni-Projekt · Synthetischer Sparkov-Datensatz · 10.000 Kunden · 2 Jahre Transaktionen"
)

st.markdown(
    """
Kreditkartenbetrug verursacht jährlich hohe Schäden, ist statistisch aber ein
seltenes Ereignis. Nur ein sehr kleiner Anteil aller Transaktionen ist
betrügerisch. Dieses Dashboard zeigt die Verteilung der Betrugsfälle in den
Daten und bewertet, wie gut verschiedene Machine-Learning-Modelle sie erkennen.
"""
)

if u.require_processed_data():
    meta = u.load_meta()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transaktionen", u.fmt_int(meta["n_transactions"]))
    c2.metric("Kreditkarten", u.fmt_int(meta["n_cards"]))
    c3.metric("Betrugsfälle", u.fmt_int(meta["n_fraud"]))
    c4.metric(
        "Betrugsrate",
        u.fmt_pct(meta["fraud_rate"]),
        help="Anteil betrügerischer Transaktionen. Die starke Klassenungleichheit "
        "ist die zentrale Herausforderung.",
    )

    st.divider()

    st.markdown(
        f"""
**Zeitraum:** {meta["ts_min"][:10]} bis {meta["ts_max"][:10]} ·
**Quelle:** {meta["n_source_files"]} CSV-Dateien (Sparkov-Generator), aufbereitet
zu Parquet. Personenbezogen wirkende Felder (Name, Adresse, SSN, Kartennummer)
wurden dabei entfernt bzw. maskiert.

**Seiten:**
1. Datensatz: Struktur, Segmente und ein Blick in die (maskierten) Rohdaten
2. Explorative Analyse: Verteilung der Betrugsfälle nach Zeit, Betrag, Händlerkategorie und Segment
3. Feature Engineering: von Rohspalten zu prädiktiven Merkmalen
4. Modellvergleich: vier Modelle und passende Metriken für seltene Ereignisse
5. Threshold und Kosten: der Schwellwert als Abwägung zwischen Treffern und Fehlalarmen
6. Case Explorer: einzelne Transaktionen und Modellfehler im Detail
"""
    )
