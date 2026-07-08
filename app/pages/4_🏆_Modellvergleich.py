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
Vier klassische Modelle und der beste LLM-Ansatz wurden auf demselben Testset
verglichen: {u.fmt_int(res['test_size'])} Transaktionen mit {res['n_fraud_test']}
Betrugsfällen, identisch für alle Ansätze. Accuracy ist hier wenig aussagekräftig. Bei einer Betrugsrate von
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

# Bestes LLM-Setup als Nebenvergleich ergaenzen (siehe Fussnote)
llm = u.load_llm_results()
llm_best = max(llm["runs"], key=lambda r: r["f1"])
llm_cm = llm_best["cm"]
llm_row = pd.DataFrame([{
    "Modell": f"LLM: {llm['model']} ({llm_best['method']}/{llm_best['view']}) *",
    "Threshold": np.nan,
    "Precision": llm_best["precision"], "Recall": llm_best["recall"], "F1": llm_best["f1"],
    "erkannt (TP)": llm_cm[1][1], "übersehen (FN)": llm_cm[1][0], "Fehlalarme (FP)": llm_cm[0][1],
}])
tbl_full = pd.concat([tbl, llm_row], ignore_index=True).sort_values(
    "F1", ascending=False, ignore_index=True
)

st.subheader("Kennzahlen der Betrugsklasse")
melt = tbl_full.melt(id_vars="Modell", value_vars=["Precision", "Recall", "F1"],
                     var_name="Metrik", value_name="Wert")
fig = px.bar(
    melt, x="Modell", y="Wert", color="Metrik", barmode="group",
    color_discrete_sequence=[COLOR_LEGIT, COLOR_FRAUD, "#9B8FD1"],
    category_orders={"Modell": tbl_full["Modell"].tolist()},
)
fig.update_layout(height=400, yaxis_range=[0, 1], yaxis_title="")
st.plotly_chart(fig, width="stretch")

st.dataframe(
    tbl_full, hide_index=True, width="stretch",
    column_config={
        "Precision": st.column_config.NumberColumn(format="%.3f"),
        "Recall": st.column_config.NumberColumn(format="%.3f"),
        "F1": st.column_config.NumberColumn(format="%.3f"),
        "Threshold": st.column_config.NumberColumn(format="%.2f"),
    },
)

st.caption(
    "* Bestes LLM-Setup aus dem LLM-Vergleich (Seite 7), ausgewertet auf demselben "
    "Testset; der Schwellwert liegt dort auf einer Score-Skala von 0 bis 100 und ist "
    "deshalb nicht angegeben."
)

st.markdown(
    f"""
Einordnung der Ergebnisse:
- {best} erreicht den höchsten F1-Wert und zugleich den höchsten Recall
  ({models['LightGBM']['recall']:.0%}), findet also die meisten Betrugsfälle.
- XGBoost ist der ausgewogenste Ansatz: Precision und Recall liegen mit je
  {models['XGBoost']['precision']:.0%} gleichauf, zugleich die höchste Precision im Feld.
- Random Forest liegt dahinter; er übersieht mehr Betrug (Recall
  {models['Random Forest']['recall']:.0%}) bei ähnlicher Precision.
- Die logistische Regression fällt deutlich ab. Auch mit hohem Schwellwert bleibt die
  Precision niedrig. Ein lineares Modell reicht für diese Muster nicht aus.

Die Gradient-Boosting-Verfahren (LightGBM, XGBoost) setzen sich durch, weil sie die
nichtlinearen Wechselwirkungen zwischen den Merkmalen (etwa hoher Betrag, späte
Uhrzeit und Abweichung vom kartenüblichen Betrag) am besten abbilden. Das beste
LLM-Setup liegt deutlich unter den Baumverfahren und nur vor der logistischen
Regression; Details dazu auf der Seite "LLM vs. klassische Modelle".
"""
)

# --- Konfusionsmatrizen ----------------------------------------------------
st.subheader("Konfusionsmatrizen")
llm_label = f"LLM: {llm['model']} ({llm_best['method']}/{llm_best['view']})"
sel = st.selectbox("Modell auswählen", tbl["Modell"].tolist() + [llm_label])
if sel == llm_label:
    cm = np.array(llm_best["cm"])
else:
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
