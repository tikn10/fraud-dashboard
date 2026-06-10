"""Seite 2: Explorative Analyse — Wo versteckt sich Fraud?"""
import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils as u  # noqa: E402
from config import COLOR_FRAUD, COLOR_LEGIT, WEEKDAYS_DE  # noqa: E402

u.page_setup("Explorative Analyse", "🔍")
st.title("🔍 Wo versteckt sich Fraud?")

if not u.require_processed_data():
    st.stop()

meta = u.load_meta()
base_rate = meta["fraud_rate"]

st.markdown(
    f"""
Die Grundrate liegt bei **{u.fmt_pct(base_rate)}** — Fraud ist also extrem selten.
Aber er ist **nicht gleichverteilt**: Bestimmte Uhrzeiten, Beträge und Kategorien
sind deutlich auffälliger. Genau diese Muster muss ein Modell später lernen.
"""
)

# ===========================================================================
# 1) Fraud-Rate nach Uhrzeit
# ===========================================================================
st.subheader("⏰ Tageszeit: Fraud schlägt nachts zu")

by_hour = u.load_agg("by_hour").sort_values("hour")
fig = go.Figure()
fig.add_bar(
    x=by_hour["hour"],
    y=by_hour["fraud_rate"] * 100,
    marker_color=[
        COLOR_FRAUD if r > 2 * base_rate else COLOR_LEGIT
        for r in by_hour["fraud_rate"]
    ],
    hovertemplate="Stunde %{x}: %{y:.2f} %<extra></extra>",
)
fig.add_hline(
    y=base_rate * 100,
    line_dash="dot",
    line_color="#8899AA",
    annotation_text=f"Ø {u.fmt_pct(base_rate)}",
)
fig.update_layout(
    xaxis_title="Stunde des Tages",
    yaxis_title="Fraud-Rate (%)",
    height=380,
    xaxis=dict(dtick=2),
)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Rot markiert: Stunden mit mehr als doppelter Durchschnittsrate. "
    "Der Generator modelliert hier ein reales Muster: Betrüger agieren bevorzugt "
    "nachts, wenn Karteninhaber schlafen und Auffälligkeiten später bemerkt werden."
)

# ===========================================================================
# 2) Betrag
# ===========================================================================
st.subheader("💰 Betrag: Fraud ist teurer")

sample = u.load_plot_sample()
log_amt = sample.assign(log_amt=np.log10(sample["amt"].clip(lower=0.01)))
fig = px.histogram(
    log_amt,
    x="log_amt",
    color="Klasse",
    histnorm="percent",
    barmode="overlay",
    opacity=0.65,
    nbins=80,
    color_discrete_map=u.FRAUD_COLOR_MAP,
    labels={"log_amt": "Betrag ($, log-Skala)"},
)
tick_vals = [0, 1, 2, 3]
fig.update_xaxes(tickvals=tick_vals, ticktext=[f"{10**v:,.0f}" for v in tick_vals])
fig.update_layout(yaxis_title="Anteil je Klasse (%)", height=380)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Verteilungen normiert je Klasse (Fraud wäre absolut sonst unsichtbar). "
    "Datenbasis: stratifiziertes Sample. Log-Skala, da Beträge stark rechtsschief sind."
)

# ===========================================================================
# 3) Kategorie
# ===========================================================================
st.subheader("🛒 Kategorie: Online-Shopping ist das Einfallstor")

by_cat = u.load_agg("by_category").sort_values("fraud_rate", ascending=True)
fig = px.bar(
    by_cat,
    x=by_cat["fraud_rate"] * 100,
    y="category",
    orientation="h",
    labels={"x": "Fraud-Rate (%)", "category": ""},
    color=by_cat["fraud_rate"],
    color_continuous_scale=[COLOR_LEGIT, COLOR_FRAUD],
)
fig.add_vline(x=base_rate * 100, line_dash="dot", line_color="#8899AA")
fig.update_layout(height=480, coloraxis_showscale=False)
st.plotly_chart(fig, width="stretch")

# ===========================================================================
# 4) Segmente
# ===========================================================================
st.subheader("👥 Segmente: Wer ist am stärksten betroffen?")

seg = u.load_agg("by_segment")
seg["Gruppe"] = seg["age_group"] + " · " + seg["area"]
piv = (
    seg.groupby(["Gruppe", "gender"], observed=True)
    .apply(lambda d: d["n_fraud"].sum() / d["n"].sum(), include_groups=False)
    .unstack()
    * 100
)
piv = piv.rename(columns={"F": "weiblich", "M": "männlich"})
fig = px.imshow(
    piv,
    text_auto=".2f",
    color_continuous_scale=["#1A2332", COLOR_FRAUD],
    labels=dict(color="Fraud-Rate (%)"),
    aspect="auto",
)
fig.update_layout(height=380, xaxis_title="", yaxis_title="")
st.plotly_chart(fig, width="stretch")

# ===========================================================================
# 5) Zeitverlauf + Wochentag
# ===========================================================================
st.subheader("📅 Verlauf über zwei Jahre")

col1, col2 = st.columns([2, 1])
with col1:
    by_month = u.load_agg("by_month").sort_values("month")
    fig = go.Figure()
    fig.add_scatter(
        x=by_month["month"],
        y=by_month["fraud_rate"] * 100,
        mode="lines+markers",
        line=dict(color=COLOR_FRAUD, width=2),
        hovertemplate="%{x}: %{y:.2f} %<extra></extra>",
    )
    fig.update_layout(
        xaxis_title="Monat", yaxis_title="Fraud-Rate (%)", height=340
    )
    st.plotly_chart(fig, width="stretch")
with col2:
    by_wd = u.load_agg("by_weekday")
    by_wd["weekday"] = by_wd["weekday"].astype(
        "category"
    ).cat.set_categories(WEEKDAYS_DE)
    by_wd = by_wd.sort_values("weekday")
    fig = px.bar(
        by_wd,
        x="weekday",
        y=by_wd["fraud_rate"] * 100,
        labels={"weekday": "", "y": "Fraud-Rate (%)"},
        color_discrete_sequence=[COLOR_LEGIT],
    )
    fig.update_layout(height=340)
    st.plotly_chart(fig, width="stretch")

# ===========================================================================
# Kritische Einordnung
# ===========================================================================
st.divider()
st.markdown(
    """
#### ⚠️ Einordnung: synthetische Daten

Die Muster oben sind **bewusst vom Generator injiziert** (Nacht-Peak, hohe
Beträge, Online-Kategorien) und daher *sauberer* als in echten Bankdaten.
Zwei Konsequenzen für unsere Arbeit:

- Gute Modellmetriken sind **teilweise ein Artefakt** der klaren Muster —
  reale Fraud-Detection ist schwerer.
- Die Distanz Kunde ↔ Händler ist hier **kein** Signal, weil der Generator
  Händler zufällig um den Kunden platziert — anders als in der Realität.
  Wer das Feature prüft und verwirft, hat die Daten verstanden.
"""
)
