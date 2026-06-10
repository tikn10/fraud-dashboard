"""Startseite: Projekt-Framing + Kernzahlen. Aufruf:  streamlit run app/Home.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as u  # noqa: E402

u.page_setup("Übersicht")

st.title("🛡️ Kreditkarten-Fraud-Detection")
st.caption(
    "Uni-Projekt · Synthetischer Sparkov-Datensatz · 10.000 Kunden · 2 Jahre Transaktionen"
)

st.markdown(
    """
Kreditkartenbetrug verursacht jährlich Milliardenschäden – ist aber statistisch
gesehen ein **Nadel-im-Heuhaufen-Problem**: Nur ein winziger Bruchteil aller
Transaktionen ist betrügerisch. Dieses Dashboard zeigt, **wo** sich Fraud in den
Daten versteckt und **wie gut** Machine-Learning-Modelle ihn finden.
"""
)

if u.require_processed_data():
    meta = u.load_meta()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transaktionen", u.fmt_int(meta["n_transactions"]))
    c2.metric("Kreditkarten", u.fmt_int(meta["n_cards"]))
    c3.metric("Fraud-Fälle", u.fmt_int(meta["n_fraud"]))
    c4.metric(
        "Fraud-Rate",
        u.fmt_pct(meta["fraud_rate"]),
        help="Anteil betrügerischer Transaktionen – die zentrale Herausforderung: "
        "extreme Klassenungleichheit.",
    )

    st.divider()

    st.markdown(
        f"""
**Zeitraum:** {meta["ts_min"][:10]} bis {meta["ts_max"][:10]} ·
**Quelle:** {meta["n_source_files"]} CSV-Dateien (Sparkov-Generator), aufbereitet
zu Parquet — personenbezogen wirkende Felder (Name, Adresse, SSN, Kartennummer)
wurden dabei entfernt bzw. maskiert.

**Seiten:**
1. **Datensatz** – Struktur, Segmente und ein Blick in die (maskierten) Rohdaten
2. **Explorative Analyse** – Wo versteckt sich Fraud? Zeit, Betrag, Kategorie, Segment
3. *Modellvergleich, Threshold-Analyse und Case Explorer folgen, sobald
   Modell-Ergebnisse vorliegen.*
"""
    )
