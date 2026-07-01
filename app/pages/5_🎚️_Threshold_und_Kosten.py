"""Seite 5: Threshold & Kosten — Fraud Detection als Abwägung, nicht als reine Klassifikation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils as u  # noqa: E402
from config import COLOR_ACCENT, COLOR_FRAUD, COLOR_GRID, COLOR_LEGIT  # noqa: E402
from config import COST_FN_DEFAULT, COST_FP_DEFAULT  # noqa: E402

u.page_setup("Threshold & Kosten", "🎚️")
st.title("🎚️ Schwellwert & Kosten")

st.markdown(
    """
Ein Modell gibt keine Ja/Nein-Antwort, sondern eine **Wahrscheinlichkeit**. Erst
der **Schwellwert** entscheidet, ab wann eine Transaktion als Betrug gilt. Diese
eine Stellschraube verändert alles – und es gibt **kein objektiv richtiges**
Optimum, sondern nur das, was zu den Kosten von Fehlern passt.

> Illustriert an einem **Random-Forest-Lauf**. Schiebt den Schwellwert und
> beobachtet, wie sich Treffer, Fehlalarme und Kosten verschieben.
"""
)

curve = u.load_threshold_curve()

# Illustrativer RF-Lauf mit vollständiger Schwellwert-Kurve (eigene, in sich
# konsistente Zahlenbasis – unabhängig vom finalen 10k-Vergleichs-Set, für das
# nur ein einzelner Arbeitspunkt je Modell vorliegt).
N_FRAUD = 944
N_LEGIT = 172_635

# --- Slider ---------------------------------------------------------------
thr = st.select_slider(
    "Schwellwert (ab welcher Fraud-Wahrscheinlichkeit wird blockiert?)",
    options=[round(x, 2) for x in curve["threshold"]],
    value=0.25,
)
row = curve.loc[curve["threshold"] == thr].iloc[0]
prec, rec, f1 = row["precision"], row["recall"], row["f1"]

# Konfusionsmatrix aus Precision/Recall rekonstruieren
tp = int(round(rec * N_FRAUD))
fn = N_FRAUD - tp
fp = int(round(tp * (1 - prec) / prec)) if prec > 0 else 0
tn = N_LEGIT - fp

c1, c2, c3 = st.columns(3)
c1.metric("Precision", u.fmt_pct(prec, 0), help="Wie viele Alarme echt sind")
c2.metric("Recall", u.fmt_pct(rec, 0), help="Wie viel Betrug gefunden wird")
c3.metric("F1", f"{f1:.2f}")

# --- Live-Konfusionsmatrix + Trade-off-Kurve ------------------------------
colA, colB = st.columns(2)
with colA:
    cm = np.array([[tn, fp], [fn, tp]])
    z_text = [[f"{v:,}".replace(",", ".") for v in r] for r in cm]
    fig = go.Figure(go.Heatmap(
        z=cm, x=["vorhergesagt: legitim", "vorhergesagt: Fraud"],
        y=["tatsächlich: legitim", "tatsächlich: Fraud"],
        text=z_text, texttemplate="%{text}", textfont={"size": 17},
        colorscale=[[0, COLOR_GRID], [1, COLOR_FRAUD]], showscale=False,
    ))
    fig.update_layout(height=340, title=f"Konfusionsmatrix bei Schwellwert {thr:.2f}")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")
with colB:
    fig = go.Figure()
    fig.add_scatter(x=curve["threshold"], y=curve["precision"], name="Precision",
                    line=dict(color=COLOR_LEGIT, width=2))
    fig.add_scatter(x=curve["threshold"], y=curve["recall"], name="Recall",
                    line=dict(color=COLOR_FRAUD, width=2))
    fig.add_vline(x=thr, line_dash="dash", line_color=COLOR_ACCENT)
    fig.update_layout(height=340, title="Precision/Recall über den Schwellwert",
                      xaxis_title="Schwellwert", yaxis_range=[0, 1.02])
    st.plotly_chart(fig, width="stretch")

st.caption(
    "Niedriger Schwellwert → mehr Betrug gefunden (hoher Recall), aber mehr Fehlalarme "
    "(niedrige Precision). Hoher Schwellwert → umgekehrt. Die Kurven kreuzen sich – "
    "dort liegt die beste Balance (F1)."
)

# --- Kostenrechnung -------------------------------------------------------
st.subheader("💶 Was kostet welcher Fehler?")
st.markdown(
    "Fraud Detection ist am Ende eine **Kostenfrage**. Ein übersehener Betrug (FN) "
    "kostet den durchgelassenen Schaden; ein Fehlalarm (FP) kostet manuelle Prüfung "
    "und Kundenärger. Stellt die Annahmen ein und findet den **kostenminimalen** "
    "Schwellwert:"
)
col1, col2 = st.columns(2)
cost_fn = col1.number_input("Kosten je übersehenem Fraud (FN) in €", 0, 5000, COST_FN_DEFAULT, 10)
cost_fp = col2.number_input("Kosten je Fehlalarm (FP) in €", 0, 500, COST_FP_DEFAULT, 1)

# Kosten über alle Schwellwerte
cost_df = curve.copy()
cost_df["tp"] = (cost_df["recall"] * N_FRAUD).round()
cost_df["fn"] = N_FRAUD - cost_df["tp"]
cost_df["fp"] = np.where(cost_df["precision"] > 0,
                         (cost_df["tp"] * (1 - cost_df["precision"]) / cost_df["precision"]).round(), 0)
cost_df["Gesamtkosten"] = cost_df["fn"] * cost_fn + cost_df["fp"] * cost_fp
best_thr = cost_df.loc[cost_df["Gesamtkosten"].idxmin(), "threshold"]
cur_cost = cost_df.loc[cost_df["threshold"] == thr, "Gesamtkosten"].iloc[0]
min_cost = cost_df["Gesamtkosten"].min()

fig = go.Figure()
fig.add_scatter(x=cost_df["threshold"], y=cost_df["Gesamtkosten"], mode="lines+markers",
                line=dict(color=COLOR_ACCENT, width=2), name="Gesamtkosten")
fig.add_vline(x=best_thr, line_dash="dot", line_color=COLOR_FRAUD,
              annotation_text=f"Kostenminimum @ {best_thr:.2f}")
fig.add_vline(x=thr, line_dash="dash", line_color=COLOR_LEGIT,
              annotation_text=f"aktuell @ {thr:.2f}")
fig.update_layout(height=360, xaxis_title="Schwellwert", yaxis_title="Gesamtkosten (€)")
st.plotly_chart(fig, width="stretch")

m1, m2, m3 = st.columns(3)
m1.metric("Kosten beim aktuellen Schwellwert", f"{cur_cost:,.0f} €".replace(",", "."))
m2.metric("Kostenminimum", f"{min_cost:,.0f} €".replace(",", "."), f"bei Schwellwert {best_thr:.2f}")
m3.metric("Einsparpotenzial", f"{cur_cost - min_cost:,.0f} €".replace(",", "."),
          delta_color="inverse")

st.info(
    "💡 **Kernaussage für die Präsentation:** Der „beste“ Schwellwert hängt von den "
    "Kostenannahmen ab, nicht vom Modell allein. Wer übersehenen Betrug teuer "
    "ansetzt, senkt den Schwellwert (mehr Recall); wer Fehlalarme scheut, hebt ihn an. "
    "Das Modell liefert die Wahrscheinlichkeiten – die Entscheidung trifft das Geschäft."
)
