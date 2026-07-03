"""Seite 3: Feature Engineering. Von Rohspalten zu prädiktiven Merkmalen."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils as u  # noqa: E402
from config import COLOR_ACCENT, COLOR_FRAUD, COLOR_LEGIT  # noqa: E402

u.page_setup("Feature Engineering", "🧬")
st.title("🧬 Feature Engineering")

st.markdown(
    """
Die Rohdaten allein sind für ein Modell wenig aussagekräftig. Der entscheidende
Schritt ist, aus Zeitstempel, Betrag und Transaktionshistorie geeignete Merkmale
abzuleiten. Diese Features bestimmen, wie gut die Modelle Betrug erkennen.
"""
)

# --- Welche Features wurden gebaut -----------------------------------------
st.subheader("Von der Rohspalte zum Merkmal")

feat = pd.DataFrame(
    [
        ("amt", "Betrag", "direkt", "Betrugstransaktionen sind im Schnitt deutlich teurer."),
        ("amt_ratio_7d", "Betrag ÷ 7-Tage-Mittel der Karte", "abgeleitet",
         "Setzt den Betrag ins Verhältnis zum üblichen Verhalten der jeweiligen Karte. Ein Betrag von 300 $ kann bei einer Karte normal sein, bei einer anderen ein Ausreißer."),
        ("hour", "Stunde aus Zeitstempel", "abgeleitet", "Bildet das nächtliche Muster des Betrugs ab."),
        ("velocity_1h", "Transaktionen pro Stunde je Karte", "abgeleitet",
         "Ein schneller Anstieg der Transaktionsfrequenz kann auf eine kompromittierte Karte hindeuten."),
        ("day_of_week", "Wochentag", "abgeleitet", "Schwaches Wochenmuster."),
        ("age", "Alter zum Transaktionszeitpunkt", "abgeleitet (aus dob)", "Demografisches Signal."),
        ("dist_km", "Distanz Kunde zu Händler (Haversine)", "abgeleitet",
         "Wirkt plausibel, ist im synthetischen Datensatz aber nicht informativ (siehe unten)."),
        ("category / gender", "kategorische Felder", "kodiert", "One-Hot- bzw. Label-Encoding für das Modell."),
    ],
    columns=["Feature", "Bedeutung", "Typ", "Bedeutung für die Vorhersage"],
)
st.dataframe(feat, hide_index=True, width="stretch")

st.caption(
    "Velocity- und Ratio-Features benötigen die Transaktionshistorie je Karte. "
    "Vor der Berechnung wird deshalb pro Karte chronologisch sortiert und ein "
    "rollierendes Zeitfenster (1 Stunde bzw. 7 Tage) verwendet."
)

# --- Feature Importance ----------------------------------------------------
st.subheader("Feature Importance")

res = u.load_model_results()
imp = res["models"]["Random Forest"]["importances"]
imp_df = (
    pd.DataFrame({"Feature": list(imp), "Importance": list(imp.values())})
    .sort_values("Importance", ascending=True)
)
top3 = imp_df.nlargest(3, "Importance")["Importance"].sum()

fig = px.bar(
    imp_df, x="Importance", y="Feature", orientation="h",
    color="Importance", color_continuous_scale=[COLOR_LEGIT, COLOR_FRAUD],
)
fig.update_layout(height=420, coloraxis_showscale=False)
st.plotly_chart(fig, width="stretch")
st.markdown(
    f"""
Die drei wichtigsten Features (`amt`, `amt_ratio_7d`, `hour`) tragen zusammen
rund {top3:.0%} der Erklärkraft. Das entspricht den Mustern aus der explorativen
Analyse: hohe Beträge, Abweichung vom kartenüblichen Betrag und späte Uhrzeit.
"""
)

# --- Die dist_km-Pointe ----------------------------------------------------
st.subheader("Distanz als nicht-informatives Merkmal")

ep = res["error_profile"]
ep_df = pd.DataFrame(
    {"Feature": ep["features"], "übersehen (FN)": ep["missed_fn"], "erkannt (TP)": ep["caught_tp"]}
)
dist_row = ep_df[ep_df["Feature"] == "dist_km"].iloc[0]

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown(
        f"""
Die Distanz zwischen Kunde und Händler wirkt wie ein klassisches Betrugssignal
(Kauf weit entfernt vom Wohnort). In diesen Daten ist sie jedoch nicht
informativ:

- Die Feature-Importance beträgt nur {imp.get("dist_km", 0):.3f} und ist damit eine der niedrigsten.
- Bei übersehenen und bei erkannten Betrugsfällen liegt sie bei rund
  {dist_row['übersehen (FN)']:.0f} km gegenüber {dist_row['erkannt (TP)']:.0f} km,
  also praktisch ohne Unterschied.

Der Grund liegt im Generator: Er platziert Händler zufällig im Umkreis jedes
Kunden. Anders als in echten Bankdaten trägt die Distanz hier keine Information.
"""
    )
with col2:
    comp = imp_df.copy()
    comp["Gruppe"] = comp["Feature"].apply(
        lambda f: "dist_km" if f == "dist_km" else ("Top 3" if f in ["amt", "amt_ratio_7d", "hour"] else "übrige")
    )
    fig = px.bar(
        comp, x="Importance", y="Feature", orientation="h", color="Gruppe",
        color_discrete_map={"Top 3": COLOR_FRAUD, "dist_km": COLOR_ACCENT, "übrige": COLOR_LEGIT},
    )
    fig.update_layout(height=420, legend_title="")
    st.plotly_chart(fig, width="stretch")
