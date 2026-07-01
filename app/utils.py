"""Gemeinsame Helfer für alle App-Seiten: Daten-Loader (gecacht), Plotly-Theme, Formatierung."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402

# ---------------------------------------------------------------------------
# Plotly-Theme: dunkles "Monitoring"-Design, passend zur Streamlit-Theme-Config
# ---------------------------------------------------------------------------
_template = go.layout.Template(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans Pro, sans-serif", color=cfg.COLOR_TEXT, size=15),
        xaxis=dict(gridcolor=cfg.COLOR_GRID, zerolinecolor=cfg.COLOR_GRID),
        yaxis=dict(gridcolor=cfg.COLOR_GRID, zerolinecolor=cfg.COLOR_GRID),
        colorway=[cfg.COLOR_LEGIT, cfg.COLOR_FRAUD, cfg.COLOR_ACCENT],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(bgcolor=cfg.COLOR_BG2),
    )
)
pio.templates["fraud_dark"] = _template
pio.templates.default = "fraud_dark"

FRAUD_COLOR_MAP = {"Fraud": cfg.COLOR_FRAUD, "legitim": cfg.COLOR_LEGIT}


# ---------------------------------------------------------------------------
# Daten-Loader (alle gecacht — die App liest nur die kompakten Artefakte)
# ---------------------------------------------------------------------------
def processed_data_available() -> bool:
    return cfg.META_PATH.exists()


@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    return json.loads(cfg.META_PATH.read_text())


@st.cache_data(show_spinner=False)
def load_agg(name: str) -> pd.DataFrame:
    """Aggregat-Tabelle laden, z. B. 'by_hour', 'by_category', 'by_segment'."""
    return pd.read_parquet(cfg.AGG_DIR / f"{name}.parquet")


@st.cache_data(show_spinner="Lade Plot-Sample …")
def load_plot_sample() -> pd.DataFrame:
    df = pd.read_parquet(cfg.PLOT_SAMPLE_PATH)
    df["Klasse"] = df["is_fraud"].map({0: "legitim", 1: "Fraud"})
    return df


@st.cache_data(show_spinner=False)
def load_model_results() -> dict:
    return json.loads(cfg.MODEL_RESULTS_PATH.read_text())


@st.cache_data(show_spinner=False)
def load_threshold_curve() -> pd.DataFrame:
    return pd.read_csv(cfg.RF_THRESHOLD_CURVE_PATH)


@st.cache_data(show_spinner=False)
def load_llm_results() -> dict:
    return json.loads(cfg.LLM_RESULTS_PATH.read_text())


@st.cache_data(show_spinner=False)
def load_rules(view: str) -> str:
    path = cfg.RULES_RAW_PATH if view == "raw" else cfg.RULES_ENGINEERED_PATH
    return path.read_text(encoding="utf-8")


def case_explorer_available() -> bool:
    return cfg.CASE_EXPLORER_PATH.exists()


@st.cache_data(show_spinner="Lade Fallbeispiele …")
def load_case_explorer() -> pd.DataFrame:
    df = pd.read_parquet(cfg.CASE_EXPLORER_PATH)
    df["Klasse"] = df["y_true"].map({0: "legitim", 1: "Fraud"})
    return df


# ---------------------------------------------------------------------------
# Formatierung & UI-Bausteine
# ---------------------------------------------------------------------------
def fmt_int(x: float) -> str:
    """1234567 -> '1.234.567' (deutsche Tausenderpunkte)."""
    return f"{int(x):,}".replace(",", ".")


def fmt_pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f} %".replace(".", ",")


def page_setup(title: str, icon: str = "🛡️") -> None:
    """Einheitlicher Seitenkopf für alle App-Seiten."""
    st.set_page_config(page_title=f"{title} · Fraud Detection", page_icon=icon, layout="wide")


def require_processed_data() -> bool:
    """Zeigt eine Anleitung, falls die Pipeline noch nicht gelaufen ist."""
    if processed_data_available():
        return True
    st.warning(
        "**Noch keine aufbereiteten Daten gefunden.**\n\n"
        "Bitte einmalig die Pipeline ausführen:\n\n"
        "```\npython scripts/01_preprocess.py\n```\n\n"
        f"Erwarteter Rohdaten-Ordner: `{cfg.RAW_DATA_DIR}`\n"
        "(anpassbar in `config.py` oder per Umgebungsvariable `FRAUD_DATA_DIR`)"
    )
    return False
