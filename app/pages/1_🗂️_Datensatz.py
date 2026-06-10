"""Seite 1: Der Datensatz — Struktur, Segmente, maskierte Beispieldaten."""
import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils as u  # noqa: E402
from config import COLOR_LEGIT  # noqa: E402

u.page_setup("Datensatz", "🗂️")
st.title("🗂️ Der Datensatz")

if not u.require_processed_data():
    st.stop()

meta = u.load_meta()

st.markdown(
    """
Der Datensatz wurde mit dem **Sparkov-Generator** erzeugt: synthetische
Kreditkartentransaktionen, deren Kundenpopulation auf realer US-Demografie
basiert. Jede Transaktion trägt ein Label `is_fraud` — wir wissen also für
jede Zeile, ob sie betrügerisch war. Das macht den Datensatz ideal zum
**Trainieren und Bewerten** von Fraud-Detection-Modellen.
"""
)

# --- Kundensegmente -------------------------------------------------------
st.subheader("Kundensegmente")
st.markdown(
    "Sparkov erzeugt Kunden nach **Profilen** (Altersgruppe × Geschlecht × "
    "Wohngegend) und simuliert je Profil unterschiedliches Kaufverhalten. "
    "So verteilen sich die Transaktionen auf die Segmente:"
)

seg = u.load_agg("by_segment")
seg["Segment"] = (
    seg["age_group"] + " · " + seg["gender"].map({"M": "männl.", "F": "weibl."}) + " · " + seg["area"]
)
seg = seg.sort_values("n", ascending=True)
fig = px.bar(
    seg,
    x="n",
    y="Segment",
    orientation="h",
    labels={"n": "Transaktionen"},
    color_discrete_sequence=[COLOR_LEGIT],
)
fig.update_layout(height=420)
st.plotly_chart(fig, width="stretch")

# --- Maskierte Beispieldaten ----------------------------------------------
st.subheader("Blick in die Daten (maskiert)")
st.markdown(
    "Felder wie Name, Adresse, SSN und Kartennummer wurden in der Aufbereitung "
    "entfernt; Karten sind nur noch als anonyme ID + letzte vier Ziffern sichtbar "
    "(*Privacy by Design* — auch wenn alle Daten synthetisch sind)."
)

sample = u.load_plot_sample()
show_cols = [
    "card_label", "ts", "category", "merchant", "amt",
    "age_group", "area", "state", "distance_km", "is_fraud",
]
preview = sample[show_cols].sample(12, random_state=7).sort_values("ts")
st.dataframe(
    preview,
    hide_index=True,
    width="stretch",
    column_config={
        "card_label": "Karte",
        "ts": st.column_config.DatetimeColumn("Zeitpunkt", format="DD.MM.YYYY HH:mm"),
        "category": "Kategorie",
        "merchant": "Händler",
        "amt": st.column_config.NumberColumn("Betrag ($)", format="%.2f"),
        "age_group": "Alter",
        "area": "Gegend",
        "state": "Staat",
        "distance_km": st.column_config.NumberColumn("Distanz (km)", format="%.0f"),
        "is_fraud": "Fraud",
    },
)

st.info(
    "ℹ️ **Hinweis zur Vorschau:** Die Tabelle stammt aus einem stratifizierten "
    "Plot-Sample, in dem Fraud-Fälle überrepräsentiert sind — die echte "
    f"Fraud-Rate liegt bei **{u.fmt_pct(meta['fraud_rate'])}**."
)
