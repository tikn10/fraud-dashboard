"""Seite 5: Threshold und Kosten. Betrugserkennung als Abwägung, nicht als reine Klassifikation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils as u  # noqa: E402
import config as cfg  # noqa: E402
from config import COLOR_ACCENT, COLOR_FRAUD, COLOR_GRID, COLOR_LEGIT  # noqa: E402
from config import COST_FP_DEFAULT  # noqa: E402

u.page_setup("Schwellenwert und Kosten", "🎚️")
st.title("🎚️ Schwellenwert und Kosten")

intro_placeholder = st.empty()

if not u.lgbm_predictions_available():
    st.warning(
        "Vorhersage-Datei noch nicht erzeugt.\n\n"
        "Einmalig lokal ausführen (greift auf die Modeling-Parquets zu):\n\n"
        "```\npip install lightgbm scikit-learn\npython scripts/03_lgbm_threshold_data.py\n```\n\n"
        "Das Skript trainiert LightGBM wie im Team-Lauf nach und schreibt "
        "`results/lgbm_eval_predictions.parquet`. Danach erscheint diese Seite automatisch."
    )
    st.stop()

@st.cache_data(show_spinner=False)
def dataset_counts() -> dict:
    df = pd.read_parquet(cfg.LGBM_EVAL_PREDICTIONS_PATH, columns=["y_true"])
    n_total = len(df)
    n_fraud = int(df["y_true"].sum())
    return {"total": n_total, "fraud": n_fraud, "legit": n_total - n_fraud}


_c = dataset_counts()
N_TOTAL, N_FRAUD, N_LEGIT = _c["total"], _c["fraud"], _c["legit"]
_n_total_str = f"{N_TOTAL:,}".replace(",", ".")

intro_placeholder.markdown(
    f"""
Ein Modell gibt keine Ja/Nein-Antwort, sondern eine Wahrscheinlichkeit. Erst der
Schwellenwert legt fest, ab wann eine Transaktion als Betrug gilt. Er hat großen
Einfluss auf das Ergebnis, und es gibt kein objektiv richtiges Optimum, sondern
nur eines, das zu den Kosten der jeweiligen Fehler passt.

Gezeigt wird LightGBM, das Modell mit dem höchsten F1-Wert. Zur besseren
statistischen Aussagekraft läuft diese Analyse auf einem größeren, im Training
ungenutzten Auswertungsset ({_n_total_str} Transaktionen, {N_FRAUD} Betrugsfälle);
der Modellvergleich bleibt davon unberührt auf dem gemeinsamen 7.000er-Testset.
Der Schwellenwert lässt sich in 0,01-Schritten verschieben; 
Treffer, Fehlalarme und Kosten ändern sich entsprechend.
"""
)


@st.cache_data(show_spinner=False)
def threshold_table() -> pd.DataFrame:
    """Vollständige Kennzahl- und Kostenbasis je Schwellenwert (0,01–0,99),
    vektorisiert und ohne Argumente, damit Streamlit nichts hashen muss.
    Enthält precision, recall, f1, tp, fp, fn und die Summe der übersehenen
    Betragssummen (fn_amount) je Schwellenwert."""
    df = pd.read_parquet(cfg.LGBM_EVAL_PREDICTIONS_PATH)
    y = df["y_true"].to_numpy()
    p = df["y_pred_proba"].to_numpy()
    a = df["amt"].to_numpy()

    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    n_fraud = int((y == 1).sum())
    fraud_amt_total = float(a[y == 1].sum())

    # Sortierung nach Wahrscheinlichkeit -> Kennzahlen per kumulativer Summe
    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]
    a_sorted = a[order]

    # Für jeden Threshold: Index, ab dem p >= t gilt (alles davor wird "legitim" vorhergesagt)
    idx = np.searchsorted(p_sorted, thresholds, side="left")

    # Kumulative Anzahl Betrug bzw. Betragssumme UNTERHALB eines Index
    cum_fraud = np.concatenate([[0], np.cumsum(y_sorted == 1)])
    cum_fraud_amt = np.concatenate([[0.0], np.cumsum(np.where(y_sorted == 1, a_sorted, 0.0))])

    fn = cum_fraud[idx]                       # übersehene Betrugsfälle (unter Schwelle)
    fn_amount = cum_fraud_amt[idx]            # deren Betragssumme
    tp = n_fraud - fn                         # erkannte Betrugsfälle
    predicted_pos = len(p) - idx              # als Betrug vorhergesagt (>= Schwelle)
    fp = predicted_pos - tp                   # Fehlalarme

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), 0.0)

    return pd.DataFrame({
        "threshold": thresholds, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp.astype(int), "fp": fp.astype(int), "fn": fn.astype(int),
        "fn_amount": fn_amount,
    })


curve = threshold_table()

# Kennpunkte der Kurve
f1_best = curve.loc[curve["f1"].idxmax()]

# --- Slider (0,01-Schritte) -------------------------------------------------
_default_thr = min(list(curve["threshold"]), key=lambda x: abs(x - u.default_threshold()))
thr = st.select_slider(
    "Schwellenwert (ab welcher Betrugswahrscheinlichkeit wird blockiert?)",
    options=list(curve["threshold"]),
    value=_default_thr,
)
row = curve.loc[curve["threshold"] == thr].iloc[0]
tp, fp, fn = int(row["tp"]), int(row["fp"]), int(row["fn"])
tn = N_LEGIT - fp

c1, c2, c3 = st.columns(3)
c1.metric("Precision", u.fmt_pct(row["precision"], 1), help="Anteil echter Alarme")
c2.metric("Recall", u.fmt_pct(row["recall"], 1), help="Anteil gefundener Betrugsfälle")
c3.metric("F1", f"{row['f1']:.3f}")

# --- Live-Konfusionsmatrix + Kurven ------------------------------------------
colA, colB = st.columns(2)
with colA:
    cm = np.array([[tn, fp], [fn, tp]])
    z_text = [[f"{v:,}".replace(",", ".") for v in r] for r in cm]
    fig = go.Figure(go.Heatmap(
        z=cm, x=["vorhergesagt: legitim", "vorhergesagt: Betrug"],
        y=["tatsächlich: legitim", "tatsächlich: Betrug"],
        text=z_text, texttemplate="%{text}", textfont={"size": 17},
        colorscale=[[0, COLOR_GRID], [1, COLOR_FRAUD]], showscale=False,
    ))
    fig.update_layout(height=340, title=f"Konfusionsmatrix bei Schwellenwert {thr:.2f}")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")
with colB:
    fig = go.Figure()
    fig.add_scatter(x=curve["threshold"], y=curve["precision"], name="Precision",
                    line=dict(color=COLOR_LEGIT, width=2))
    fig.add_scatter(x=curve["threshold"], y=curve["recall"], name="Recall",
                    line=dict(color=COLOR_FRAUD, width=2))
    fig.add_vline(x=float(f1_best["threshold"]), line_dash="dot", line_color="#9B8FD1",
                  annotation_text=f"F1-Maximum ({f1_best['threshold']:.2f})")
    fig.add_vline(x=thr, line_dash="dash", line_color=COLOR_ACCENT)
    fig.update_layout(height=360, title="Precision und Recall über den Schwellenwert",
                      xaxis_title="Schwellenwert", yaxis_range=[0, 1.02],
                      legend=dict(orientation="h", yanchor="top", y=-0.2))
    st.plotly_chart(fig, width="stretch")

st.caption(
    f"Ein niedriger Schwellenwert findet mehr Betrug (hoher Recall), erzeugt aber mehr "
    f"Fehlalarme (niedrige Precision). Ein hoher Schwellenwert wirkt umgekehrt. Der "
    f"beste Kompromiss nach F1 liegt bei {f1_best['threshold']:.2f} "
    f"(F1 {f1_best['f1']:.3f}); der wirtschaftlich beste Punkt folgt unten aus der "
    f"Kostenbetrachtung. Die Standardposition des Reglers entspricht dem Arbeitspunkt "
    f"des Teams auf der kalibrierten Skala."
)

# --- Kostenbetrachtung --------------------------------------------------------
st.subheader("Kostenbetrachtung")
st.markdown(
    """
Die Wahl des Schwellenwerts ist letztlich eine Kostenfrage. Für übersehenen Betrug
(FN) stehen zwei Rechenweisen zur Auswahl: der tatsächliche Transaktionsbetrag
(die FN-Kosten sind dann die Summe der Beträge aller übersehenen Betrugsfälle)
oder ein fiktiver Pauschalbetrag je Fall. Ein Fehlalarm (FP) verursacht in beiden
Varianten einen pauschalen Aufwand für manuelle Prüfung und Kundenkontakt.
"""
)
fn_mode = st.radio(
    "Kostenbasis für übersehenen Betrug (FN)",
    ["Tatsächlicher Transaktionsbetrag", "Pauschalbetrag je Fall"],
    horizontal=True,
)
col_a, col_b = st.columns(2)
with col_a:
    cost_fp = st.number_input("Kosten je Fehlalarm (FP) in $", 0, 500, COST_FP_DEFAULT, 1)
with col_b:
    cost_fn_flat = st.number_input(
        "Pauschale je übersehenem Betrug (FN) in $", 0, 5000, 200, 10,
        disabled=(fn_mode == "Tatsächlicher Transaktionsbetrag"),
        help="Wird nur in der Pauschal-Variante verwendet.",
    )


cost_df = curve.copy()
if fn_mode == "Tatsächlicher Transaktionsbetrag":
    cost_df["Gesamtkosten"] = cost_df["fn_amount"] + cost_df["fp"] * cost_fp
else:
    cost_df["Gesamtkosten"] = cost_df["fn"] * cost_fn_flat + cost_df["fp"] * cost_fp

best_row = cost_df.loc[cost_df["Gesamtkosten"].idxmin()]
best_thr = float(best_row["threshold"])
cur_cost = float(cost_df.loc[cost_df["threshold"] == thr, "Gesamtkosten"].iloc[0])
min_cost = float(best_row["Gesamtkosten"])

fig = go.Figure()
fig.add_scatter(x=cost_df["threshold"], y=cost_df["Gesamtkosten"], mode="lines",
                line=dict(color=COLOR_ACCENT, width=2), name="Gesamtkosten")
fig.add_vline(x=best_thr, line_dash="dot", line_color=COLOR_FRAUD,
              annotation_text=f"Kostenminimum ({best_thr:.2f})")
fig.add_vline(x=thr, line_dash="dash", line_color=COLOR_LEGIT,
              annotation_text=f"aktuell ({thr:.2f})")
fig.update_layout(height=360, xaxis_title="Schwellenwert",
                  yaxis_title="Gesamtkosten ($) im Auswertungsset")
st.plotly_chart(fig, width="stretch")

m1, m2, m3 = st.columns(3)
m1.metric("Kosten beim aktuellen Schwellenwert", f"{cur_cost:,.0f} $".replace(",", "."))
m2.metric("Kostenminimum", f"{min_cost:,.0f} $".replace(",", "."),
          f"bei Schwellenwert {best_thr:.2f}")
m3.metric("Einsparpotenzial", f"{cur_cost - min_cost:,.0f} $".replace(",", "."),
          delta_color="inverse")

if fn_mode == "Tatsächlicher Transaktionsbetrag":
    st.info(
        "Der kostenminimale Schwellenwert hängt von den Kostenannahmen ab, nicht vom "
        "Modell allein. Weil übersehene Betrugsfälle hier mit ihrem tatsächlichen "
        "Betrag zu Buche schlagen, wiegen wenige teure Fälle schwerer als viele "
        "kleine. Werden Fehlalarme stärker gewichtet, steigt der optimale "
        "Schwellenwert. Alle Beträge beziehen sich auf das Auswertungsset."
    )
else:
    st.info(
        "In der Pauschal-Variante zählt jeder übersehene Betrugsfall gleich viel, "
        "unabhängig vom Transaktionsbetrag. Das Kostenminimum verschiebt sich mit "
        "dem Verhältnis der beiden Pauschalen: Je teurer ein übersehener Fall "
        "angesetzt wird, desto niedriger der optimale Schwellenwert. Alle Beträge "
        "beziehen sich auf das Auswertungsset."
    )
