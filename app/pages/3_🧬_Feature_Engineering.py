"""Seite 3: Feature Engineering — von Rohspalten zu prädiktiven Merkmalen."""
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
Die Rohdaten allein sind für ein Modell schwach. Der entscheidende Schritt ist,
aus Zeitstempel, Betrag und Historie **aussagekräftige Merkmale** abzuleiten.
Genau hier steckt das Datenverständnis – und genau diese Features bestimmen
später, wie gut die Modelle Fraud erkennen.
"""
)

# --- Welche Features wurden gebaut -----------------------------------------
st.subheader("Von der Rohspalte zum Merkmal")

feat = pd.DataFrame(
    [
        ("amt", "Betrag", "direkt", "Betrugstransaktionen sind im Schnitt deutlich teurer."),
        ("amt_ratio_7d", "Betrag ÷ 7-Tage-Mittel der Karte", "abgeleitet",
         "Setzt den Betrag ins Verhältnis zum normalen Verhalten DIESER Karte – ein 300-$-Kauf ist bei der einen Karte normal, bei der anderen ein Ausreißer."),
        ("hour", "Stunde aus Zeitstempel", "abgeleitet", "Fängt das Nacht-Muster des Betrugs ein."),
        ("velocity_1h", "Transaktionen pro Stunde je Karte", "abgeleitet",
         "Schneller Schlagzahl-Anstieg deutet auf eine kompromittierte Karte hin."),
        ("day_of_week", "Wochentag", "abgeleitet", "Schwaches Saison-/Wochenmuster."),
        ("age", "Alter zum Transaktionszeitpunkt", "abgeleitet (aus dob)", "Demografisches Signal."),
        ("dist_km", "Distanz Kunde ↔ Händler (Haversine)", "abgeleitet",
         "Klingt plausibel – ist im synthetischen Datensatz aber WIRKUNGSLOS (siehe unten)."),
        ("category / gender", "kategorische Felder", "kodiert", "One-Hot- bzw. Label-Encoding fürs Modell."),
    ],
    columns=["Feature", "Bedeutung", "Typ", "Warum es zählt"],
)
st.dataframe(feat, hide_index=True, width="stretch")

st.caption(
    "Velocity- und Ratio-Features brauchen die Transaktions­historie je Karte – "
    "deshalb wird vor der Berechnung pro Karte chronologisch sortiert und ein "
    "rollierendes Zeitfenster (1 h bzw. 7 Tage) verwendet."
)

# --- Feature Importance ----------------------------------------------------
st.subheader("Was das beste Modell wirklich nutzt")

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
st.success(
    f"**Drei Features tragen rund {top3:.0%} der Erklärkraft:** `amt`, `amt_ratio_7d`, "
    "`hour`. Genau die Muster, die wir in der explorativen Analyse gesehen haben – "
    "hohe Beträge, Ausreißer vom Kartennormal, Nachtstunden. Das Modell bestätigt die EDA."
)

# --- Die dist_km-Pointe ----------------------------------------------------
st.subheader("⚠️ Ein Feature, das nichts bringt – und warum das wichtig ist")

ep = res["error_profile"]
ep_df = pd.DataFrame(
    {"Feature": ep["features"], "übersehen (FN)": ep["missed_fn"], "erkannt (TP)": ep["caught_tp"]}
)
dist_row = ep_df[ep_df["Feature"] == "dist_km"].iloc[0]

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown(
        f"""
Die **Distanz Kunde ↔ Händler** wirkt wie ein klassisches Betrugssignal
(„Kauf weit weg vom Wohnort“). In diesen Daten ist sie aber **wertlos**:

- Feature-Importance nur **{imp.get("dist_km", 0):.3f}** (eines der schwächsten Features)
- Bei übersehenen *und* erkannten Frauds liegt sie bei rund
  **{dist_row['übersehen (FN)']:.0f} km** vs. **{dist_row['erkannt (TP)']:.0f} km** –
  praktisch **kein Unterschied**

Der Grund liegt im Generator: Er platziert Händler **zufällig im Umkreis**
jedes Kunden. Anders als in echten Bankdaten trägt die Distanz daher keine
Information. Das Feature zu bauen, zu prüfen und dann **bewusst als
wirkungslos einzuordnen**, zeigt, dass wir die Daten verstanden haben.
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
