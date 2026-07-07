"""Seite 5: Threshold und Kosten. Betrugserkennung als Abwägung, nicht als reine Klassifikation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils as u  # noqa: E402
from config import COLOR_ACCENT, COLOR_FRAUD, COLOR_GRID, COLOR_LEGIT  # noqa: E402
from config import COST_FP_DEFAULT  # noqa: E402

u.page_setup("Schwellwert und Kosten", "🎚️")
st.title("🎚️ Schwellwert und Kosten")

st.markdown(
    """
Ein Modell gibt keine Ja/Nein-Antwort, sondern eine Wahrscheinlichkeit. Erst der
Schwellwert legt fest, ab wann eine Transaktion als Betrug gilt. Er hat großen
Einfluss auf das Ergebnis, und es gibt kein objektiv richtiges Optimum, sondern
nur eines, das zu den Kosten der jeweiligen Fehler passt.

Gezeigt wird LightGBM, das Modell mit dem höchsten F1-Wert, auf dem
10.000er-Eval-Set aus dem Modellvergleich. Der Schwellwert lässt sich in
0,01-Schritten verschieben; Treffer, Fehlalarme und Kosten ändern sich
entsprechend.
"""
)

if not u.lgbm_predictions_available():
    st.warning(
        "Vorhersage-Datei noch nicht erzeugt.\n\n"
        "Einmalig lokal ausführen (greift auf die Modeling-Parquets zu):\n\n"
        "```\npip install lightgbm scikit-learn\npython scripts/03_lgbm_threshold_data.py\n```\n\n"
        "Das Skript trainiert LightGBM wie im Team-Lauf nach und schreibt "
        "`results/lgbm_eval_predictions.parquet`. Danach erscheint diese Seite automatisch."
    )
    st.stop()

pred_df = u.load_lgbm_predictions()
y_true = pred_df["y_true"].to_numpy()
proba = pred_df["y_pred_proba"].to_numpy()
amt = pred_df["amt"].to_numpy()

N_TOTAL = len(pred_df)
N_FRAUD = int(y_true.sum())
N_LEGIT = N_TOTAL - N_FRAUD


@st.cache_data(show_spinner=False)
def sweep_curve(y: tuple, p: tuple) -> pd.DataFrame:
    """Precision/Recall/F1 für Schwellwerte 0,01 bis 0,99 (Schrittweite 0,01)."""
    y = np.asarray(y)
    p = np.asarray(p)
    rows = []
    for t in np.round(np.arange(0.01, 1.00, 0.01), 2):
        pred = p >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"threshold": t, "precision": prec, "recall": rec, "f1": f1,
                     "tp": tp, "fp": fp, "fn": fn})
    return pd.DataFrame(rows)


curve = sweep_curve(tuple(y_true), tuple(proba))

# Kennpunkte der Kurve
f1_best = curve.loc[curve["f1"].idxmax()]
pr_cross = curve.loc[(curve["precision"] - curve["recall"]).abs().idxmin()]

# --- Slider (0,01-Schritte) -------------------------------------------------
thr = st.select_slider(
    "Schwellwert (ab welcher Betrugswahrscheinlichkeit wird blockiert?)",
    options=list(curve["threshold"]),
    value=0.65,
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
    fig.update_layout(height=340, title=f"Konfusionsmatrix bei Schwellwert {thr:.2f}")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")
with colB:
    fig = go.Figure()
    fig.add_scatter(x=curve["threshold"], y=curve["precision"], name="Precision",
                    line=dict(color=COLOR_LEGIT, width=2))
    fig.add_scatter(x=curve["threshold"], y=curve["recall"], name="Recall",
                    line=dict(color=COLOR_FRAUD, width=2))
    fig.add_scatter(x=[pr_cross["threshold"]], y=[pr_cross["precision"]],
                    mode="markers", name="Precision = Recall",
                    marker=dict(size=11, color="#E8EDF5", symbol="circle-open",
                                line=dict(width=2)))
    fig.add_vline(x=float(f1_best["threshold"]), line_dash="dot", line_color="#9B8FD1",
                  annotation_text=f"F1-Maximum ({f1_best['threshold']:.2f})")
    fig.add_vline(x=thr, line_dash="dash", line_color=COLOR_ACCENT)
    fig.update_layout(height=340, title="Precision und Recall über den Schwellwert",
                      xaxis_title="Schwellwert", yaxis_range=[0, 1.02])
    st.plotly_chart(fig, width="stretch")

st.caption(
    f"Ein niedriger Schwellwert findet mehr Betrug (hoher Recall), erzeugt aber mehr "
    f"Fehlalarme (niedrige Precision). Ein hoher Schwellwert wirkt umgekehrt. Der "
    f"markierte Punkt (Precision = Recall, bei {pr_cross['threshold']:.2f}) ist der "
    f"Schnittpunkt beider Kurven; er ist eine mögliche Balance, aber kein Optimum. Der "
    f"beste Kompromiss nach F1 liegt bei {f1_best['threshold']:.2f} "
    f"(F1 {f1_best['f1']:.3f}); der wirtschaftlich beste Punkt folgt unten aus der "
    f"Kostenbetrachtung. Der Arbeitspunkt des Teams (0,65) ist die Standardposition "
    f"des Reglers."
)

# --- Kostenbetrachtung --------------------------------------------------------
st.subheader("Kostenbetrachtung")
st.markdown(
    """
Die Wahl des Schwellwerts ist letztlich eine Kostenfrage. Für übersehenen Betrug
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


@st.cache_data(show_spinner=False)
def fn_amount_by_threshold(y: tuple, p: tuple, a: tuple) -> pd.DataFrame:
    """Summe der Beträge übersehener Betrugsfälle je Schwellwert."""
    y = np.asarray(y); p = np.asarray(p); a = np.asarray(a)
    fraud_mask = y == 1
    rows = []
    for t in np.round(np.arange(0.01, 1.00, 0.01), 2):
        missed = fraud_mask & (p < t)
        rows.append({"threshold": t, "fn_amount": float(a[missed].sum())})
    return pd.DataFrame(rows)


fn_amt = fn_amount_by_threshold(tuple(y_true), tuple(proba), tuple(amt))
cost_df = curve.merge(fn_amt, on="threshold")
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
fig.update_layout(height=360, xaxis_title="Schwellwert",
                  yaxis_title="Gesamtkosten ($) im 10.000er-Testset")
st.plotly_chart(fig, width="stretch")

m1, m2, m3 = st.columns(3)
m1.metric("Kosten beim aktuellen Schwellwert", f"{cur_cost:,.0f} $".replace(",", "."))
m2.metric("Kostenminimum", f"{min_cost:,.0f} $".replace(",", "."),
          f"bei Schwellwert {best_thr:.2f}")
m3.metric("Einsparpotenzial", f"{cur_cost - min_cost:,.0f} $".replace(",", "."),
          delta_color="inverse")

if fn_mode == "Tatsächlicher Transaktionsbetrag":
    st.info(
        "Der kostenminimale Schwellwert hängt von den Kostenannahmen ab, nicht vom "
        "Modell allein. Weil übersehene Betrugsfälle hier mit ihrem tatsächlichen "
        "Betrag zu Buche schlagen, wiegen wenige teure Fälle schwerer als viele "
        "kleine. Werden Fehlalarme stärker gewichtet, steigt der optimale "
        "Schwellwert. Alle Beträge beziehen sich auf das 10.000er-Testset."
    )
else:
    st.info(
        "In der Pauschal-Variante zählt jeder übersehene Betrugsfall gleich viel, "
        "unabhängig vom Transaktionsbetrag. Das Kostenminimum verschiebt sich mit "
        "dem Verhältnis der beiden Pauschalen: Je teurer ein übersehener Fall "
        "angesetzt wird, desto niedriger der optimale Schwellwert. Alle Beträge "
        "beziehen sich auf das 10.000er-Testset."
    )
