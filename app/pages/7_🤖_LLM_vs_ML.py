"""Seite 7: LLM vs. klassische Modelle — kann ein Sprachmodell Fraud erkennen?"""
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
Als Erweiterung haben wir getestet, ob ein **Sprachmodell (Claude Haiku 4.5)**
Kreditkartenbetrug erkennen kann – ohne klassisches Training, nur aus den
Transaktionsmerkmalen im Prompt. Drei Ansätze, jeweils auf **rohen** und auf
**aufbereiteten** Merkmalen:

- **Hardlabel** – das Modell gibt direkt ein Urteil (Betrug: ja/nein).
- **Confidence** – das Modell gibt einen Score 0–100, Schwellwert wird optimiert.
- **Rules** – das Modell leitet zuerst aus Beispielen eigene Regeln ab und wendet sie dann an.

Die Frage: Kommt ein LLM an die trainierten Modelle heran?
"""
)

ml = u.load_model_results()
llm = u.load_llm_results()

# --- Gemeinsamer F1-Vergleich ---------------------------------------------
st.subheader("Die Antwort in einem Bild")

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
    labels={"F1": "F1-Score (Fraud-Klasse)"},
)
fig.update_layout(height=460, legend_title="", xaxis_range=[0, 0.85])
st.plotly_chart(fig, width="stretch")

best_ml = max(ml["models"].items(), key=lambda kv: kv[1]["f1"])
best_llm = max(llm["runs"], key=lambda r: r["f1"])
st.error(
    f"**Die klassischen Modelle gewinnen klar.** Das beste ML-Modell "
    f"({best_ml[0]}, F1 {best_ml[1]['f1']:.2f}) liegt deutlich über dem besten "
    f"LLM-Ansatz ({best_llm['method']}/{best_llm['view']}, F1 {best_llm['f1']:.2f}). "
    "Nur die logistische Regression wird vom besten LLM-Setup geschlagen."
)

# --- Precision/Recall-Landkarte -------------------------------------------
st.subheader("Warum die LLMs scheitern: die Präzisions-Falle")
st.markdown(
    "Jeder Punkt ist ein Ansatz. **Oben rechts** ist gut (viel Betrug gefunden *und* "
    "wenige Fehlalarme). Die ML-Modelle sitzen dort. Die LLMs kleben am **linken Rand** – "
    "sie erkennen zwar teils Betrug, aber um den Preis massenhafter Fehlalarme."
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
    "Blau = klassisches ML, orange (Raute) = LLM. Das extremste Beispiel: "
    "der Hardlabel-Ansatz auf Rohdaten erkennt 100 % des Betrugs (rechts unten), "
    "markiert dafür aber fast **alles** als Betrug – Precision nahe null. Praktisch unbrauchbar."
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
**Muster, die sich zeigen:**

- **Regeln schlagen alles.** Wenn das LLM erst eigene Indikatoren ableitet und
  dann danach urteilt, wird es am besten – Struktur hilft.
- **Ohne Struktur überschätzt es Betrug massiv.** Der direkte Ja/Nein-Ansatz
  ignoriert die 0,5-%-Grundrate und ruft viel zu oft „Betrug“.
- **Laufzeit & Kosten:** Die LLM-Läufe dauerten Minuten bis über eine halbe
  Stunde und kosten API-Gebühren. Die trainierten Modelle liefern dasselbe
  in **Sekunden**.
"""
    )

# --- Die abgeleiteten Regeln + Halluzination ------------------------------
st.subheader("🔎 Das Aha-Erlebnis: Was das LLM über Betrug „glaubt“")
st.markdown(
    "Beim Rules-Ansatz hat das LLM aus Beispielen selbst Betrugsindikatoren "
    "formuliert. Die Liste ist lesbar und plausibel – enthält aber einen "
    "aufschlussreichen **Fehler**:"
)
st.info(
    "Das LLM behauptet in den Rohdaten-Regeln, die **geografische Distanz** "
    "Kunde↔Händler (> 100 km) sei ein *„zuverlässiger Indikator“* für Betrug. "
    "Das ist eine Intuition aus der **echten Welt** – in unseren synthetischen "
    "Daten aber **nachweislich falsch**: Der Generator platziert Händler zufällig "
    "um den Kunden, und `dist_km` hat in allen trainierten Modellen eine der "
    "**niedrigsten** Feature-Importances. Das LLM hat also ein Muster halluziniert, "
    "das die Bäume korrekt als wertlos erkannt haben."
)
with st.expander("📄 Vom LLM abgeleitete Regeln ansehen (Roh- vs. aufbereitete Sicht)"):
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

Für **hochvolumiges, strukturiertes** Transaktions-Scoring ist ein spezialisiertes
ML-Modell dem Sprachmodell klar überlegen – schneller, günstiger und deutlich
treffsicherer. Der Wert des LLM liegt woanders: Es kann in **lesbaren Regeln**
erklären, *warum* etwas verdächtig wirkt – und macht dabei sogar die menschlichen
Denkfehler sichtbar (siehe Distanz). Als erklärende Ergänzung interessant, als
Klassifikator für diese Aufgabe nicht.
"""
)

st.caption(
    "Methodischer Hinweis: Die LLM-Läufe wurden auf 7.000 Test-Zeilen (38 Fraud) "
    "ausgewertet, die klassischen Modelle auf dem umgebenden 10.000er-Set (55 Fraud). "
    "Beide teilen dieselbe Grundrate (~0,55 %); für einen bit-genauen Kopf-an-Kopf-"
    "Vergleich müssten sie auf identischen Zeilen laufen."
)
