"""Seite 4: Modellvergleich. Vier Modelle und passende Metriken."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils as u  # noqa: E402
from config import COLOR_FRAUD, COLOR_GRID, COLOR_LEGIT  # noqa: E402

u.page_setup("Modellvergleich", "🏆")
st.title("🏆 Modellvergleich")

res = u.load_model_results()
models = res["models"]

st.markdown(
    f"""
Vier Modelle wurden auf denselben {u.fmt_int(res['test_size'])} Test-Transaktionen
verglichen. Accuracy ist hier wenig aussagekräftig. Bei einer Betrugsrate von
{u.fmt_pct(__import__('config').CANONICAL_FRAUD_RATE)} erreicht bereits ein Modell,
das jede Transaktion als legitim einstuft, über 99 % Accuracy, ohne einen
einzigen Betrugsfall zu erkennen. Aussagekräftig sind Precision und Recall der
Betrugsklasse.
"""
)

# --- Metrik-Tabelle + Balken ----------------------------------------------
rows = []
for name, m in models.items():
    cm = m["cm"]
    rows.append({
        "Modell": name, "Threshold": m["threshold"],
        "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"],
        "erkannt (TP)": cm[1][1], "übersehen (FN)": cm[1][0], "Fehlalarme (FP)": cm[0][1],
    })
tbl = pd.DataFrame(rows).sort_values("F1", ascending=False)
best = tbl.iloc[0]["Modell"]

st.subheader("Kennzahlen der Betrugsklasse")
melt = tbl.melt(id_vars="Modell", value_vars=["Precision", "Recall", "F1"],
                var_name="Metrik", value_name="Wert")
fig = px.bar(
    melt, x="Modell", y="Wert", color="Metrik", barmode="group",
    color_discrete_sequence=[COLOR_LEGIT, COLOR_FRAUD, "#9B8FD1"],
    category_orders={"Modell": tbl["Modell"].tolist()},
)
fig.update_layout(height=400, yaxis_range=[0, 1], yaxis_title="")
st.plotly_chart(fig, width="stretch")

st.dataframe(
    tbl, hide_index=True, width="stretch",
    column_config={
        "Precision": st.column_config.NumberColumn(format="%.3f"),
        "Recall": st.column_config.NumberColumn(format="%.3f"),
        "F1": st.column_config.NumberColumn(format="%.3f"),
        "Threshold": st.column_config.NumberColumn(format="%.2f"),
    },
)

st.markdown(
    f"""
Einordnung der Ergebnisse:
- {best} erreicht den höchsten F1-Wert und damit die beste Balance aus Precision und Recall.
- XGBoost liegt nahezu gleichauf und hat den höchsten Recall
  ({models['XGBoost']['recall']:.0%}), erkennt also die meisten Betrugsfälle.
- Random Forest hat die höchste Precision ({models['Random Forest']['precision']:.0%},
  also wenige Fehlalarme), übersieht dafür mehr Betrug (Recall
  {models['Random Forest']['recall']:.0%}).
- Die logistische Regression fällt deutlich ab. Auch mit hohem Schwellwert bleibt die
  Precision niedrig. Ein lineares Modell reicht für diese Muster nicht aus.

Die Gradient-Boosting-Verfahren (LightGBM, XGBoost) setzen sich durch, weil sie die
nichtlinearen Wechselwirkungen zwischen den Merkmalen (etwa hoher Betrag, späte
Uhrzeit und Abweichung vom kartenüblichen Betrag) am besten abbilden.
"""
)

# --- Konfusionsmatrizen ----------------------------------------------------
st.subheader("Konfusionsmatrizen")
sel = st.selectbox("Modell auswählen", tbl["Modell"].tolist())
cm = np.array(models[sel]["cm"])

labels = ["legitim", "Betrug"]
z_text = [[f"{v:,}".replace(",", ".") for v in row] for row in cm]
fig = go.Figure(go.Heatmap(
    z=cm, x=[f"vorhergesagt: {l}" for l in labels], y=[f"tatsächlich: {l}" for l in labels],
    text=z_text, texttemplate="%{text}", textfont={"size": 18},
    colorscale=[[0, COLOR_GRID], [1, COLOR_FRAUD]], showscale=False,
))
fig.update_layout(height=360)
fig.update_yaxes(autorange="reversed")
st.plotly_chart(fig, width="stretch")

tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
c1, c2, c3 = st.columns(3)
c1.metric("Erkannt (TP)", u.fmt_int(tp), help="Betrug korrekt gefunden")
c2.metric("Übersehen (FN)", u.fmt_int(fn), help="Betrug durchgelassen (der teure Fehler)")
c3.metric("Fehlalarme (FP)", u.fmt_int(fp), help="Legitime Transaktion fälschlich blockiert")

st.divider()
st.caption(
    "Hinweis zur Methodik: Der Threshold wurde auf einem separaten Validierungs-Split "
    "gewählt, nicht auf dem Testset. Der Train/Test-Split erfolgte zufällig und nicht "
    "zeitbasiert. Ein zeitbasierter Split wäre näher am realen Einsatz."
)
