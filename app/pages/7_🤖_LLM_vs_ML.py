"""Seite 7: LLM vs. klassische Modelle."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils as u  # noqa: E402
from config import COLOR_ACCENT, COLOR_FRAUD, COLOR_GRID, COLOR_LEGIT  # noqa: E402

u.page_setup("LLM vs. ML", "🤖")
st.title("🤖 LLM vs. klassische Modelle")

st.markdown(
    """
Als Erweiterung wurde getestet, ob ein Sprachmodell (Claude Haiku 4.5)
Kreditkartenbetrug erkennen kann, ohne klassisches Training und nur anhand der
Transaktionsmerkmale im Prompt. Untersucht wurden drei Ansätze, jeweils auf
rohen und auf aufbereiteten Merkmalen:

- Hardlabel: Das Modell gibt direkt ein Urteil ab (Betrug: ja/nein).
- Confidence: Das Modell gibt einen Score von 0 bis 100 aus, der Schwellwert wird optimiert.
- Rules: Das Modell leitet zunächst aus Beispielen eigene Regeln ab und wendet sie dann an.

Die Leitfrage ist, ob ein LLM an die trainierten Modelle heranreicht.
"""
)

ml = u.load_model_results()
llm = u.load_llm_results()

# --- Gemeinsamer F1-Vergleich ---------------------------------------------
st.subheader("Vergleich der F1-Werte")

rows = []
for name, m in ml["models"].items():
    rows.append({"Ansatz": name, "Typ": "Klassisches ML", "F1": m["f1"],
                 "Precision": m["precision"], "Recall": m["recall"]})
for r in llm["runs"]:
    rows.append({"Ansatz": f"LLM · {r['method']} · {r['view']}", "Typ": "LLM (Haiku 4.5)",
                 "F1": r["f1"], "Precision": r["precision"], "Recall": r["recall"]})
df = pd.DataFrame(rows).sort_values("F1", ascending=True)

fig = px.bar(
    df, x="F1", y="Ansatz", orientation="h", color="Typ",
    color_discrete_map={"Klassisches ML": COLOR_LEGIT, "LLM (Haiku 4.5)": COLOR_ACCENT},
    labels={"F1": "F1-Score (Betrugsklasse)"},
)
fig.update_layout(height=460, legend_title="", xaxis_range=[0, 0.85])
st.plotly_chart(fig, width="stretch")

best_ml = max(ml["models"].items(), key=lambda kv: kv[1]["f1"])
best_llm = max(llm["runs"], key=lambda r: r["f1"])
st.markdown(
    f"""
Die klassischen Modelle schneiden deutlich besser ab. Das beste ML-Modell
({best_ml[0]}, F1 {best_ml[1]['f1']:.2f}) erreicht einen höheren F1-Wert als der
beste LLM-Ansatz ({best_llm['method']}/{best_llm['view']}, F1 {best_llm['f1']:.2f}).
Nur die logistische Regression wird vom besten LLM-Ansatz übertroffen.
"""
)

# --- Precision/Recall-Landkarte -------------------------------------------
st.subheader("Precision und Recall im Vergleich")
st.markdown(
    "Jeder Punkt steht für einen Ansatz. Oben rechts ist der günstige Bereich (viel "
    "Betrug erkannt und zugleich wenige Fehlalarme). Dort liegen die klassischen "
    "Modelle. Die LLM-Ansätze liegen am linken Rand: Sie erkennen teils Betrug, "
    "erzeugen dabei aber sehr viele Fehlalarme."
)
fig = go.Figure()
for name, m in ml["models"].items():
    fig.add_scatter(x=[m["recall"]], y=[m["precision"]], mode="markers+text",
                    text=[name], textposition="top center", name=name,
                    marker=dict(size=14, color=COLOR_LEGIT, line=dict(width=1, color="white")),
                    showlegend=False)
for r in llm["runs"]:
    fig.add_scatter(x=[r["recall"]], y=[r["precision"]], mode="markers+text",
                    text=[f"{r['method']}/{r['view']}"], textposition="top center",
                    marker=dict(size=12, color=COLOR_ACCENT, symbol="diamond"),
                    showlegend=False)
fig.update_layout(
    height=460, xaxis_title="Recall (Anteil gefundener Betrugsfälle)",
    yaxis_title="Precision (Anteil echter Alarme)",
    xaxis_range=[-0.02, 1.05], yaxis_range=[-0.02, 1.05],
)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Blau steht für klassisches ML, orange (Raute) für die LLM-Ansätze. Ein "
    "auffälliges Beispiel ist der Hardlabel-Ansatz auf Rohdaten: Er erkennt 100 % "
    "des Betrugs, markiert dafür aber nahezu alle Transaktionen als Betrug, sodass "
    "die Precision nahe null liegt."
)

# --- Was hat am besten funktioniert? --------------------------------------
st.subheader("Innerhalb der LLM-Ansätze")
llm_df = pd.DataFrame(llm["runs"])
llm_df["Ansatz"] = llm_df["method"] + " · " + llm_df["view"]
c1, c2 = st.columns([3, 2])
with c1:
    melt = llm_df.melt(id_vars="Ansatz", value_vars=["precision", "recall", "f1"],
                       var_name="Metrik", value_name="Wert")
    fig = px.bar(melt, x="Ansatz", y="Wert", color="Metrik", barmode="group",
                 color_discrete_sequence=[COLOR_LEGIT, COLOR_FRAUD, "#9B8FD1"])
    fig.update_layout(height=360, xaxis_tickangle=-30, yaxis_title="", legend_title="")
    st.plotly_chart(fig, width="stretch")
with c2:
    st.markdown(
        """
Beobachtungen:

- Der Rules-Ansatz schneidet am besten ab. Wenn das LLM zunächst eigene
  Indikatoren ableitet und danach urteilt, steigt die Trefferquote.
- Ohne diese Struktur überschätzt das Modell den Betrugsanteil deutlich. Der
  direkte Ja/Nein-Ansatz berücksichtigt die Grundrate von etwa 0,5 % kaum und
  stuft zu viele Transaktionen als Betrug ein.
- Laufzeit und Kosten: Die LLM-Läufe dauerten Minuten bis über eine halbe Stunde
  und verursachen API-Kosten. Die trainierten Modelle liefern das Ergebnis in Sekunden.
"""
    )

# --- Die abgeleiteten Regeln + Halluzination ------------------------------
st.subheader("Vom Sprachmodell abgeleitete Regeln")
st.markdown(
    "Beim Rules-Ansatz hat das LLM aus Beispielen selbst Betrugsindikatoren "
    "formuliert. Die Liste ist lesbar und plausibel, enthält aber einen "
    "aufschlussreichen Fehler."
)
st.info(
    "In den Rohdaten-Regeln nennt das LLM die geografische Distanz zwischen Kunde "
    "und Händler (> 100 km) als zuverlässigen Indikator für Betrug. Das entspricht "
    "einer Intuition aus realen Daten, ist in diesem synthetischen Datensatz aber "
    "nicht zutreffend: Der Generator platziert Händler zufällig im Umkreis des "
    "Kunden, und dist_km hat in allen trainierten Modellen eine der niedrigsten "
    "Feature-Importances. Das LLM nennt hier also ein Merkmal, das die trainierten "
    "Modelle korrekt als nicht informativ einstufen."
)
with st.expander("Vom LLM abgeleitete Regeln ansehen (Roh- vs. aufbereitete Sicht)"):
    tab1, tab2 = st.tabs(["Rohdaten-Sicht", "Aufbereitete Sicht"])
    with tab1:
        st.markdown(u.load_rules("raw"))
    with tab2:
        st.markdown(u.load_rules("engineered"))

# --- Fazit ----------------------------------------------------------------
st.divider()
st.markdown(
    """
#### Fazit

Für hochvolumiges, strukturiertes Transaktions-Scoring ist ein spezialisiertes
ML-Modell dem Sprachmodell überlegen: schneller, günstiger und treffsicherer. Der
Nutzen des LLM liegt an anderer Stelle. Es kann in lesbaren Regeln beschreiben,
warum eine Transaktion verdächtig wirkt, und macht dabei auch typische Fehlannahmen
sichtbar (siehe Distanz). Als erklärende Ergänzung ist es nützlich, als
Klassifikator für diese Aufgabe jedoch nicht geeignet.
"""
)

st.caption(
    "Methodischer Hinweis: LLM-Läufe und klassische Modelle wurden auf dem identischen "
    "7.000er-Testset (38 Betrugsfälle) ausgewertet; die Threshold-Wahl erfolgte jeweils "
    "auf dem gemeinsamen 3.000er-Validierungsset. Der Vergleich ist damit ein exakter "
    "Kopf-an-Kopf-Vergleich auf denselben Zeilen."
)
