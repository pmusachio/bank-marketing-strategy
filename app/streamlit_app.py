"""Streamlit dashboard for bank customer segments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
CLUSTERS_PATH = ROOT / "reports" / "cluster_assignments.csv"
METRICS_PATH = ROOT / "reports" / "metrics.json"


st.set_page_config(page_title="Bank Marketing Strategy", layout="wide")
st.title("Bank Marketing Strategy")

if not CLUSTERS_PATH.exists():
    st.info("Run `PYTHONPATH=src python -m bank_marketing_strategy.cli train` to generate reports/cluster_assignments.csv.")
    st.stop()

df = pd.read_csv(CLUSTERS_PATH)
if "cluster" not in df.columns:
    st.error("The file reports/cluster_assignments.csv must contain a `cluster` column.")
    st.stop()

metrics = {}
if METRICS_PATH.exists():
    with METRICS_PATH.open(encoding="utf-8") as fp:
        metrics = json.load(fp)

overview, profiles, records = st.tabs(["Overview", "Profiles", "Records"])

with overview:
    counts = df["cluster"].value_counts().sort_index().rename_axis("cluster").reset_index(name="customers")
    share = counts["customers"] / counts["customers"].sum()
    counts["share"] = share.map(lambda value: f"{value:.1%}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Segments", df["cluster"].nunique())
    c3.metric("Silhouette", f"{metrics.get('silhouette', 0):.3f}" if "silhouette" in metrics else "n/a")

    st.plotly_chart(px.bar(counts, x="cluster", y="customers", text="share"), use_container_width=True)
    st.dataframe(counts, use_container_width=True, hide_index=True)

with profiles:
    numeric_cols = [col for col in df.select_dtypes("number").columns if col != "cluster"]
    selected = st.multiselect("Metrics", numeric_cols, default=numeric_cols[: min(8, len(numeric_cols))])
    if selected:
        profile = df.groupby("cluster", as_index=False)[selected].mean()
        st.dataframe(profile, use_container_width=True, hide_index=True)
        melted = profile.melt(id_vars="cluster", var_name="metric", value_name="average")
        st.plotly_chart(px.bar(melted, x="cluster", y="average", color="metric", barmode="group"), use_container_width=True)
    else:
        st.warning("Select at least one numeric metric.")

with records:
    clusters = sorted(df["cluster"].dropna().unique().tolist())
    selected_clusters = st.multiselect("Segments", clusters, default=clusters)
    st.dataframe(df[df["cluster"].isin(selected_clusters)], use_container_width=True, hide_index=True)
