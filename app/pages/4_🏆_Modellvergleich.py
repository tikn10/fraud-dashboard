"""Seite 4: Modellvergleich — vier Modelle, die richtigen Metriken."""
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
verglichen. **Accuracy spielt hier keine Rolle:** Bei
{u.fmt_pct(__import__('config').CANONICAL_FRAUD_RATE)} Fraud erreicht schon ein
Modell, das *immer* „kein Betrug“ sagt, über 99 % Accuracy – und findet keinen
einzigen Betrugsfall. Was zählt, sind **Precision und Recall der Fraud-Klasse**.
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

st.subheader("Kennzahlen der Fraud-Klasse")
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
**Wie man das liest:**
- **{best}** gewinnt beim **F1** – die beste Balance aus Präzision und Trefferquote.
- **XGBoost** liegt fast gleichauf und hat den höchsten **Recall**
  ({models['XGBoost']['recall']:.0%}), fängt also die meisten Betrugsfälle.
- **Random Forest** ist am **präzisesten** ({models['Random Forest']['precision']:.0%} –
  kaum Fehlalarme), übersieht dafür aber mehr Betrug (Recall nur
  {models['Random Forest']['recall']:.0%}).
- **Logistische Regression** fällt klar ab: Selbst mit hohem Schwellwert bleibt die
  Precision niedrig. Ein lineares Modell reicht für diese Muster nicht.

Die Gradient-Boosting-Verfahren (LightGBM, XGBoost) setzen sich also durch – sie
modellieren die nichtlinearen Wechselwirkungen (hoher Betrag *und* nachts *und*
Ausreißer vom Kartennormal) am besten.
"""
)

# --- Konfusionsmatrizen ----------------------------------------------------
st.subheader("Konfusionsmatrizen")
sel = st.selectbox("Modell auswählen", tbl["Modell"].tolist())
cm = np.array(models[sel]["cm"])

labels = ["legitim", "Fraud"]
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
c2.metric("Übersehen (FN)", u.fmt_int(fn), help="Betrug durchgelassen – der teure Fehler")
c3.metric("Fehlalarme (FP)", u.fmt_int(fp), help="Legitime Transaktion fälschlich blockiert")

st.divider()
st.caption(
    "Hinweis zur Methodik: zufälliger (nicht zeitbasierter) Train/Test-Split und "
    "Threshold-Optimierung auf dem Testset. Die Werte sind daher als **Obergrenze** "
    "zu lesen – ein zeitbasierter Split wäre näher am Realeinsatz."
)
