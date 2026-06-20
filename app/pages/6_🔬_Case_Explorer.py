"""Seite 6: Case Explorer — einzelne Transaktionen und Modellfehler verstehen."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils as u  # noqa: E402
from config import COLOR_ACCENT, COLOR_FRAUD, COLOR_LEGIT  # noqa: E402

u.page_setup("Case Explorer", "🔬")
st.title("🔬 Case Explorer")

st.markdown(
    """
Aggregierte Metriken zeigen, *wie gut* ein Modell ist – aber nicht, *wo* es
scheitert. Hier schauen wir einzelne Transaktionen an: Was hat das Modell
vorhergesagt, was war es wirklich, und **warum geht es manchmal daneben?**
"""
)

if not u.case_explorer_available():
    st.warning(
        "**Fallbeispiele noch nicht erzeugt.**\n\n"
        "Einmalig lokal ausführen (greift auf das Modeling-Parquet zu):\n\n"
        "```\npip install scikit-learn\npython scripts/02_train_and_predict.py\n```\n\n"
        "Das Skript trainiert den Random Forest nach, berechnet die Wahrscheinlichkeit "
        "pro Test-Transaktion und schreibt `data/processed/case_explorer.parquet`. "
        "Danach erscheint diese Seite automatisch."
    )
    st.stop()

df = u.load_case_explorer()
THR = 0.25  # RF-Arbeitspunkt

df["pred"] = (df["y_pred_proba"] >= THR).astype(int)
df["Ergebnis"] = np.select(
    [(df.y_true == 1) & (df.pred == 1), (df.y_true == 1) & (df.pred == 0),
     (df.y_true == 0) & (df.pred == 1)],
    ["Erkannt (TP)", "Übersehen (FN)", "Fehlalarm (FP)"], default="Korrekt legitim (TN)",
)

# --- Fehlertypen-Übersicht ------------------------------------------------
st.subheader("Wo liegt das Modell richtig – und wo nicht?")
counts = df["Ergebnis"].value_counts()
cols = st.columns(4)
for col, key, color in zip(
    cols,
    ["Erkannt (TP)", "Übersehen (FN)", "Fehlalarm (FP)", "Korrekt legitim (TN)"],
    [COLOR_FRAUD, COLOR_ACCENT, "#9B8FD1", COLOR_LEGIT],
):
    col.metric(key, u.fmt_int(counts.get(key, 0)))

st.caption(
    f"Arbeitspunkt: Schwellwert {THR:.2f} (RF). Stichprobe des Testsets — alle "
    "Fraud-Fälle plus eine Auswahl legitimer Transaktionen."
)

# --- Betrag vs. Wahrscheinlichkeit ----------------------------------------
st.subheader("Der teure blinde Fleck: kleine Frauds")
st.markdown(
    "Jeder Punkt ist eine Fraud-Transaktion. Links die **übersehenen**, rechts die "
    "**erkannten**. Das Muster aus den Logs wird hier sichtbar: Das Modell findet "
    "die **großen** Beträge zuverlässig und lässt eher **kleine** durchrutschen."
)
fraud_only = df[df.y_true == 1].copy()
fig = px.scatter(
    fraud_only, x="amt", y="y_pred_proba",
    color=fraud_only["pred"].map({1: "erkannt", 0: "übersehen"}),
    color_discrete_map={"erkannt": COLOR_FRAUD, "übersehen": COLOR_ACCENT},
    labels={"amt": "Betrag ($)", "y_pred_proba": "Fraud-Wahrscheinlichkeit", "color": ""},
    hover_data=["category", "hour"] if "category" in df.columns else None,
)
fig.add_hline(y=THR, line_dash="dash", line_color="#8899AA",
              annotation_text=f"Schwellwert {THR:.2f}")
fig.update_layout(height=420)
st.plotly_chart(fig, width="stretch")

# --- Interaktiver Drill-down ----------------------------------------------
st.subheader("Einzelfälle durchsehen")
focus = st.radio(
    "Welche Fälle interessieren?",
    ["Übersehen (FN)", "Fehlalarm (FP)", "Erkannt (TP)", "Alle Fraud-Fälle"],
    horizontal=True,
)
if focus == "Alle Fraud-Fälle":
    view = df[df.y_true == 1]
else:
    view = df[df["Ergebnis"] == focus]

view = view.sort_values("amt", ascending=False)
display_cols = [c for c in ["amt", "category", "hour", "amt_ratio_7d", "velocity_1h",
                            "dist_km", "age", "y_pred_proba", "Ergebnis"] if c in view.columns]
st.dataframe(
    view[display_cols].head(50), hide_index=True, width="stretch",
    column_config={
        "amt": st.column_config.NumberColumn("Betrag ($)", format="%.2f"),
        "category": "Kategorie", "hour": "Stunde",
        "amt_ratio_7d": st.column_config.NumberColumn("Betrag/7T-Mittel", format="%.1f"),
        "velocity_1h": st.column_config.NumberColumn("Txn/h", format="%.0f"),
        "dist_km": st.column_config.NumberColumn("Distanz (km)", format="%.0f"),
        "age": "Alter",
        "y_pred_proba": st.column_config.ProgressColumn(
            "Fraud-Wahrsch.", format="%.2f", min_value=0, max_value=1),
    },
)

st.info(
    "💡 **Für die Präsentation:** Greift hier gezielt einen *übersehenen* (FN) und "
    "einen *Fehlalarm* (FP) heraus und diskutiert, warum das Modell dort danebenliegt. "
    "Fehleranalyse zu zeigen wirkt souveräner als nur Erfolge."
)
