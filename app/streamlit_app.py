"""Interactive customer-segmentation dashboard.

Assigns a credit-card customer to a behavioural segment and shows where they fall
in the segment map, with the segment profile and a suggested marketing action.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.predict import Predictor  # noqa: E402

D = config.DRACULA
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.markdown(
    f"""<style>
    .stApp {{ background-color: {D['background']}; color: {D['foreground']}; }}
    section[data-testid="stSidebar"] {{ background-color: {D['current_line']}; }}
    h1, h2, h3 {{ color: {D['purple']}; }}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor()


def style_axes(ax):
    ax.set_facecolor(D["background"])
    for s in ax.spines.values():
        s.set_color(D["current_line"])
    ax.tick_params(colors=D["foreground"])
    ax.xaxis.label.set_color(D["foreground"])
    ax.yaxis.label.set_color(D["foreground"])
    ax.grid(True, color=D["current_line"], linestyle="--", alpha=0.4)


def segment_map(predictor, point, point_segment):
    coords = np.array(predictor.pca_sample["coords"])
    labels = np.array(predictor.pca_sample["labels"])
    fig, ax = plt.subplots(figsize=(6, 3.8), facecolor=D["background"])
    for seg in sorted(set(labels)):
        m = labels == seg
        ax.scatter(coords[m, 0], coords[m, 1], s=8, alpha=0.45,
                   color=config.SEGMENT_COLORS[seg % len(config.SEGMENT_COLORS)],
                   label=f"Segment {seg}")
    ax.scatter([point[0]], [point[1]], s=220, marker="*", color=D["foreground"],
               edgecolors=config.SEGMENT_COLORS[point_segment % len(config.SEGMENT_COLORS)],
               linewidths=2, zorder=5, label="This customer")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.legend(facecolor=D["current_line"], edgecolor=D["comment"], labelcolor=D["foreground"], fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    return fig


def main():
    try:
        predictor = load_predictor()
    except FileNotFoundError:
        st.error("Model artifact not found. Run the pipeline before launching the app.")
        return

    st.title("Bank Marketing Strategy — Customer Segmentation")
    st.markdown(
        "Assigns a credit-card customer to a behavioural segment to guide profile-based marketing. "
        "Built with KMeans on standardized usage features."
    )

    with st.sidebar:
        st.header("Customer profile")
        balance = st.number_input("Balance", 0.0, 20000.0, 1500.0, 100.0)
        purchases = st.number_input("Purchases", 0.0, 50000.0, 1000.0, 100.0)
        cash_advance = st.number_input("Cash advance", 0.0, 50000.0, 200.0, 100.0)
        credit_limit = st.number_input("Credit limit", 50.0, 30000.0, 4500.0, 100.0)
        payments = st.number_input("Payments", 0.0, 50000.0, 1500.0, 100.0)
        purchases_freq = st.slider("Purchases frequency", 0.0, 1.0, 0.5, 0.05)
        prc_full = st.slider("Share of full payments", 0.0, 1.0, 0.2, 0.05)
        run = st.button("Assign segment", type="primary")

    customer = {"BALANCE": balance, "PURCHASES": purchases, "CASH_ADVANCE": cash_advance,
                "CREDIT_LIMIT": credit_limit, "PAYMENTS": payments,
                "PURCHASES_FREQUENCY": purchases_freq, "PRC_FULL_PAYMENT": prc_full}

    if run:
        res = predictor.assign(customer)
        st.subheader("Assigned segment")
        c = st.columns(3)
        c[0].metric("Segment", f"#{res['segment']}")
        c[1].metric("Profile", res["label"])
        c[2].metric("Share of base", f"{res['share_pct']:.0f}%")
        left, right = st.columns([3, 2])
        with left:
            st.pyplot(segment_map(predictor, res["coords"], res["segment"]))
        with right:
            st.markdown("**Segment average profile**")
            prof = pd.DataFrame([{"Feature": k, "Segment mean": round(v, 1)}
                                 for k, v in res["profile_means"].items()])
            st.dataframe(prof, hide_index=True, width="stretch")

    st.subheader("Segments overview")
    st.dataframe(pd.DataFrame(predictor.segment_table()).rename(
        columns={"segment": "Segment", "share_pct": "Share %", "profile": "Profile"}),
        hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
