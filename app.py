"""
E-Commerce Customer Segmentation - Streamlit frontend

Run with:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "Mall_Customers.csv"
DATA_URL = "https://raw.githubusercontent.com/abcom-mltutorials/mall/main/Mall_Customers.csv"
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
K = 5  # chosen with the Elbow Method in the notebook

INK = "#1C1C1E"
MUTED = "#6B6B70"
LINE = "#E4E1DA"
NAVY = "#1F2A44"
GOLD = "#B08D57"

# Segment names and marketing strategies, based on the same income/spending
# rules used in the notebook's "Customer Segment Analysis" cell.
SEGMENTS = {
    "high_high": {
        "name": "Premium Shoppers",
        "rule": "High income, high spending",
        "color": "#B08D57",
        "strategy": "VIP loyalty programme, early access to new launches, premium bundles and personalised recommendations.",
    },
    "high_low": {
        "name": "Careful Affluent",
        "rule": "High income, low spending",
        "color": "#2F4A6D",
        "strategy": "Quality-focused messaging, detailed product reviews, free trials and targeted personalised offers.",
    },
    "low_high": {
        "name": "Impulsive Spenders",
        "rule": "Lower income, high spending",
        "color": "#B5654A",
        "strategy": "Flash sales, Buy Now Pay Later options, trending products and limited-time deals via app notifications.",
    },
    "low_low": {
        "name": "Budget Conscious",
        "rule": "Lower income, low spending",
        "color": "#6F8466",
        "strategy": "Discount coupons, value packs, free-shipping thresholds and price-drop alerts.",
    },
    "mid": {
        "name": "Standard Customers",
        "rule": "Moderate income and spending",
        "color": "#8A8F98",
        "strategy": "Regular newsletters, cross-selling, seasonal campaigns and reward points to move them upward.",
    },
}


def segment_key(income: float, spending: float) -> str:
    if income >= 60 and spending >= 60:
        return "high_high"
    if income >= 60 and spending < 40:
        return "high_low"
    if income < 40 and spending >= 60:
        return "low_high"
    if income < 40 and spending < 40:
        return "low_low"
    return "mid"


def html(markup: str) -> None:
    """Render raw HTML; lines are stripped so Markdown never treats them as code blocks."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines() if line.strip()), unsafe_allow_html=True)


def style(fig, height=420, legend=True):
    fig.update_layout(
        height=height,
        template="none",
        font=dict(family="Inter, sans-serif", size=13, color=INK),
        title=dict(font=dict(family="Inter, sans-serif", size=14, color=MUTED), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40 if fig.layout.title.text else 10, b=10),
        showlegend=legend,
        legend=dict(orientation="h", y=-0.18, x=0, title=None, font=dict(size=12, color=MUTED)),
        hoverlabel=dict(bgcolor="white", bordercolor=LINE, font=dict(family="Inter, sans-serif", color=INK)),
    )
    axis = dict(gridcolor="#ECEAE4", zeroline=False, linecolor=LINE, tickfont=dict(color=MUTED),
                title_font=dict(color=MUTED, size=12))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


# ----------------------------------------------------------------------------
# Data + model
# ----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.read_csv(DATA_URL)


@st.cache_data
def elbow_and_silhouette(X_scaled: np.ndarray):
    rows = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_) if k > 1 else np.nan
        rows.append({"K": k, "WCSS": km.inertia_, "Silhouette": sil})
    return pd.DataFrame(rows)


@st.cache_resource
def train_model(df: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    model = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X_scaled)
    return scaler, model, X_scaled


# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide", initial_sidebar_state="collapsed")

html(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="st-"], .stMarkdown, p, li, label, input, button {{font-family: 'Inter', sans-serif;}}
    header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"], [data-testid="stDecoration"] {{display: none !important;}}
    .block-container {{max-width: 1180px; padding: 2.6rem 2rem 4rem;}}
    h1, h2, h3 {{font-family: 'Instrument Serif', serif !important; font-weight: 400 !important; letter-spacing: -0.01em; color: {INK};}}
    .stMarkdown a, a {{color: {NAVY} !important; text-decoration: none; border-bottom: 1px solid {GOLD};}}

    .brand {{display:flex; justify-content:space-between; align-items:baseline; border-bottom:1px solid {LINE}; padding-bottom:1rem; margin-bottom:2.4rem;}}
    .brand .mark {{font-family:'Instrument Serif',serif; font-size:1.45rem; color:{INK};}}
    .brand .meta {{font-size:.78rem; letter-spacing:.14em; text-transform:uppercase; color:{MUTED};}}

    .eyebrow {{font-size:.74rem; letter-spacing:.18em; text-transform:uppercase; color:{GOLD}; font-weight:600; margin-bottom:.9rem;}}
    .hero-title {{font-family:'Instrument Serif',serif; font-size:clamp(2.6rem, 6vw, 4.4rem); line-height:1.02; color:{INK}; margin:0 0 1.2rem; max-width:14ch;}}
    .hero-title em {{color:{GOLD};}}
    .lead {{font-size:1.08rem; line-height:1.7; color:#3F3F44; max-width:60ch; margin:0;}}

    .stats {{display:grid; grid-template-columns:repeat(4,1fr); border-top:1px solid {LINE}; border-bottom:1px solid {LINE}; margin:2.6rem 0 1rem;}}
    .stat {{padding:1.4rem 1.2rem; border-left:1px solid {LINE};}}
    .stat:first-child {{border-left:none; padding-left:0;}}
    .stat .v {{font-family:'Instrument Serif',serif; font-size:2.5rem; line-height:1; color:{INK};}}
    .stat .l {{font-size:.74rem; letter-spacing:.12em; text-transform:uppercase; color:{MUTED}; margin-top:.55rem;}}

    .section-label {{font-size:.74rem; letter-spacing:.16em; text-transform:uppercase; color:{MUTED}; font-weight:600; margin:2.4rem 0 .4rem;}}
    .section-title {{font-family:'Instrument Serif',serif; font-size:2rem; color:{INK}; margin:0 0 .4rem;}}
    .section-sub {{color:{MUTED}; font-size:.95rem; line-height:1.6; margin:0 0 1.2rem; max-width:70ch;}}

    .steps {{display:grid; grid-template-columns:repeat(4,1fr); gap:0; border-top:1px solid {LINE};}}
    .step {{padding:1.2rem 1.2rem 1.2rem 0; border-bottom:1px solid {LINE};}}
    .step .n {{font-family:'Instrument Serif',serif; font-size:1.3rem; color:{GOLD};}}
    .step .t {{font-weight:600; font-size:.92rem; margin:.3rem 0 .25rem; color:{INK};}}
    .step .d {{font-size:.85rem; color:{MUTED}; line-height:1.5;}}

    .share {{display:flex; height:10px; border-radius:6px; overflow:hidden; margin:.6rem 0 2rem;}}
    .seg-table {{border-top:1px solid {INK};}}
    .seg-row {{display:grid; grid-template-columns:2.3fr .9fr 1fr 1fr .8fr 3fr; gap:1.2rem; align-items:start;
               padding:1.25rem 0; border-bottom:1px solid {LINE};}}
    .seg-row.head {{padding:.8rem 0; font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; color:{MUTED}; font-weight:600;}}
    .seg-name {{display:flex; gap:.8rem; align-items:flex-start;}}
    .swatch {{width:10px; height:10px; border-radius:50%; margin-top:.45rem; flex:none;}}
    .seg-name .t {{font-family:'Instrument Serif',serif; font-size:1.35rem; line-height:1.2; color:{INK};}}
    .seg-name .s {{font-size:.8rem; color:{MUTED}; margin-top:.2rem;}}
    .num {{font-size:1.05rem; color:{INK}; font-variant-numeric:tabular-nums;}}
    .num small {{display:block; font-size:.76rem; color:{MUTED};}}
    .strategy {{font-size:.88rem; line-height:1.55; color:#3F3F44;}}

    .result {{background:{NAVY}; color:#F4F1EA; border-radius:4px; padding:2rem 2.2rem;}}
    .result .eyebrow {{color:#D9BF8F;}}
    .result .name {{font-family:'Instrument Serif',serif; font-size:2.6rem; line-height:1.05; color:#fff;}}
    .result .rule {{color:#B9BFCC; font-size:.9rem; margin-top:.4rem;}}
    .result hr {{border:none; border-top:1px solid rgba(255,255,255,.15); margin:1.4rem 0;}}
    .result .k {{font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:#B9BFCC; margin-bottom:.4rem;}}
    .result p {{color:#F4F1EA; line-height:1.6; margin:0;}}
    .result .facts {{display:flex; gap:2.4rem; margin-top:1.4rem;}}
    .result .facts .v {{font-family:'Instrument Serif',serif; font-size:1.7rem; color:#fff;}}

    .note {{font-size:.82rem; color:{MUTED}; line-height:1.6;}}
    .foot {{margin-top:4rem; padding-top:1.2rem; border-top:1px solid {LINE}; display:flex; justify-content:space-between; font-size:.8rem; color:{MUTED}; flex-wrap:wrap; gap:.6rem;}}

    .stTabs [data-baseweb="tab-list"] {{gap:2rem; border-bottom:1px solid {LINE};}}
    .stTabs [data-baseweb="tab"] {{padding:.7rem 0; font-size:.86rem; letter-spacing:.04em; color:{MUTED}; background:transparent;}}
    .stTabs [aria-selected="true"] {{color:{INK} !important; font-weight:600;}}
    .stTabs [data-baseweb="tab-highlight"] {{background:{INK}; height:2px;}}
    .stTabs [data-baseweb="tab-panel"] {{padding-top:1.6rem;}}
    [data-testid="stVerticalBlockBorderWrapper"] {{border-radius:4px !important; border-color:{LINE} !important; background:#fff;}}
    .stDownloadButton button, .stButton button {{border-radius:2px; border:1px solid {INK}; background:transparent; color:{INK}; font-weight:500;}}
    .stDownloadButton button:hover, .stButton button:hover {{background:{INK}; color:#fff; border-color:{INK};}}

    @media (max-width: 800px) {{
      .stats, .steps {{grid-template-columns:repeat(2,1fr);}}
      .stat:nth-child(3) {{border-left:none; padding-left:0;}}
      .seg-row {{grid-template-columns:repeat(4,1fr); gap:.8rem;}}
      .seg-row .seg-name, .seg-row .strategy {{grid-column:1 / -1;}}
      .seg-row.head {{display:none;}}
    }}
    </style>
    """
)

df_raw = load_data()
scaler, model, X_scaled = train_model(df_raw)
df = df_raw.copy()
df["Cluster"] = model.labels_
labels = {c: segment_key(g[FEATURES[0]].mean(), g[FEATURES[1]].mean()) for c, g in df.groupby("Cluster")}
df["Segment"] = df["Cluster"].map(lambda c: SEGMENTS[labels[c]]["name"])
cmap = {s["name"]: s["color"] for s in SEGMENTS.values()}
sil = silhouette_score(X_scaled, model.labels_)

summary = df.groupby("Cluster").agg(
    Customers=("CustomerID", "count"),
    Avg_Age=("Age", "mean"),
    Avg_Income=(FEATURES[0], "mean"),
    Avg_Spending=(FEATURES[1], "mean"),
    Female_pct=("Gender", lambda s: (s == "Female").mean() * 100),
).round(1)
summary.insert(0, "Segment", [SEGMENTS[labels[c]]["name"] for c in summary.index])
summary = summary.sort_values("Avg_Spending", ascending=False)
seg_order = list(summary["Segment"])

html(
    """
    <div class="brand">
      <div class="mark">Customer Segmentation</div>
      <div class="meta">AI &amp; ML Project &middot; K-Means Clustering</div>
    </div>
    """
)

tabs = st.tabs(["Overview", "Data", "Model", "Segments", "Predict", "Batch"])

# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------
with tabs[0]:
    html(
        f"""
        <div class="eyebrow">E-Commerce Analytics</div>
        <div class="hero-title">Understanding customers, <em>one segment</em> at a time.</div>
        <p class="lead">An unsupervised learning study that groups shoppers by annual income and spending
        behaviour using K-Means clustering, turning raw customer records into five clear segments
        with a targeted marketing strategy for each.</p>
        <div class="stats">
          <div class="stat"><div class="v">{len(df)}</div><div class="l">Customers analysed</div></div>
          <div class="stat"><div class="v">{K}</div><div class="l">Segments identified</div></div>
          <div class="stat"><div class="v">{sil:.3f}</div><div class="l">Silhouette score</div></div>
          <div class="stat"><div class="v">{len(FEATURES)}</div><div class="l">Clustering features</div></div>
        </div>
        """
    )

    left, right = st.columns([3, 2], gap="large")
    with left:
        html('<div class="section-label">Result</div><div class="section-title">The segment map</div>')
        fig = px.scatter(df, x=FEATURES[0], y=FEATURES[1], color="Segment", color_discrete_map=cmap,
                         category_orders={"Segment": seg_order}, hover_data=["CustomerID", "Gender", "Age"])
        fig.update_traces(marker=dict(size=9, opacity=0.9, line=dict(width=0.8, color="white")))
        centers = scaler.inverse_transform(model.cluster_centers_)
        fig.add_trace(go.Scatter(x=centers[:, 0], y=centers[:, 1], mode="markers", name="Centroid",
                                 marker=dict(symbol="cross-thin", size=16, line=dict(width=2, color=INK))))
        st.plotly_chart(style(fig, 440), width="stretch")
    with right:
        html(
            f"""
            <div class="section-label">Approach</div><div class="section-title">How it works</div>
            <p class="section-sub">Each customer is described by two numbers: annual income and a
            spending score from 1 to 100. After scaling, K-Means places five centroids and assigns every
            customer to the nearest one. Customers in the same group behave alike, so each group can be
            marketed to differently.</p>
            <p class="note">Dataset: <a href="https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python">
            Mall Customer Segmentation (Kaggle)</a><br>
            Source code: <a href="https://github.com/Padmashree06/5TH_Sem_AI_ML_Project">GitHub repository</a></p>
            """
        )

    steps = [
        ("Load", "200 customer records with gender, age, income and spending score."),
        ("Explore", "Distributions, gender split and feature relationships."),
        ("Scale", "Income and spending score standardised with StandardScaler."),
        ("Choose K", "Elbow Method and Silhouette score point to five clusters."),
        ("Train", "K-Means with five clusters and a fixed random state."),
        ("Evaluate", f"Silhouette score of {sil:.3f} indicates well-separated groups."),
        ("Interpret", "Each cluster named by its income and spending profile."),
        ("Act", "A marketing strategy recommended for every segment."),
    ]
    html(
        '<div class="section-label">Methodology</div><div class="section-title">Project pipeline</div>'
        '<div class="steps">'
        + "".join(f'<div class="step"><div class="n">{i:02d}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
                  for i, (t, d) in enumerate(steps, 1))
        + "</div>"
    )

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
with tabs[1]:
    n_f = int((df_raw["Gender"] == "Female").sum())
    html(
        f"""
        <div class="section-label">Exploratory analysis</div><div class="section-title">The dataset</div>
        <p class="section-sub">Mall Customer Segmentation data: {len(df_raw)} customers, {df_raw.shape[1]} attributes,
        no missing values.</p>
        <div class="stats" style="margin-top:1rem">
          <div class="stat"><div class="v">{df_raw.shape[0]}</div><div class="l">Rows</div></div>
          <div class="stat"><div class="v">{df_raw.shape[1]}</div><div class="l">Columns</div></div>
          <div class="stat"><div class="v">{int(df_raw.isnull().sum().sum())}</div><div class="l">Missing values</div></div>
          <div class="stat"><div class="v">{n_f / len(df_raw):.0%}</div><div class="l">Female customers</div></div>
        </div>
        """
    )
    st.dataframe(df_raw, width="stretch", height=300, hide_index=True)

    html('<div class="section-label">Summary</div><div class="section-title">Descriptive statistics</div>')
    st.dataframe(df_raw.drop(columns="CustomerID").describe().round(2), width="stretch")

    html('<div class="section-label">Distributions</div><div class="section-title">How the features are spread</div>')
    c1, c2, c3 = st.columns(3, gap="medium")
    for col, feat in zip((c1, c2, c3), ["Age", FEATURES[0], FEATURES[1]]):
        with col:
            fig = px.histogram(df_raw, x=feat, nbins=15, color_discrete_sequence=[NAVY], title=feat)
            fig.update_traces(marker_line_color="#FAF9F6", marker_line_width=1.5)
            fig.update_layout(yaxis_title="Customers", xaxis_title=None, bargap=0.04)
            st.plotly_chart(style(fig, 300, legend=False), width="stretch")

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        fig = px.scatter(df_raw, x=FEATURES[0], y=FEATURES[1], color="Gender", hover_data=["Age"],
                         title="Annual income vs spending score", color_discrete_map={"Female": GOLD, "Male": NAVY})
        fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.6, color="white")))
        st.plotly_chart(style(fig, 400), width="stretch")
        html('<p class="note">Five visible groups in this plot are the first hint that K = 5 is a natural choice.</p>')
    with c2:
        corr = df_raw.drop(columns=["CustomerID", "Gender"]).corr().round(2)
        corr.columns = corr.index = ["Age", "Income", "Spending"]
        fig = px.imshow(corr, text_auto=True, zmin=-1, zmax=1, title="Correlation",
                        color_continuous_scale=[[0, "#2F4A6D"], [0.5, "#FAF9F6"], [1, GOLD]])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style(fig, 400, legend=False), width="stretch")

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
with tabs[2]:
    metrics = elbow_and_silhouette(X_scaled)
    html(
        f"""
        <div class="section-label">Model selection</div><div class="section-title">Why five clusters</div>
        <p class="section-sub">K-Means needs the number of clusters in advance. K was tested from 1 to 10 using two
        complementary measures; both agree on K = {K}.</p>
        """
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = px.line(metrics, x="K", y="WCSS", markers=True, title="Elbow Method: within-cluster sum of squares")
        fig.update_traces(line=dict(color=NAVY, width=2), marker=dict(size=8, color=NAVY))
        fig.add_vline(x=K, line_dash="dot", line_color=GOLD)
        fig.add_annotation(x=K, y=metrics.loc[metrics.K == K, "WCSS"].iloc[0], text="Elbow at K = 5",
                           showarrow=False, xshift=55, yshift=18, font=dict(color=GOLD))
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(style(fig, 380, legend=False), width="stretch")
        html('<p class="note">WCSS falls as K grows. The point where the curve flattens, the elbow, '
             'marks the best trade-off between compact clusters and simplicity.</p>')
    with c2:
        m = metrics.dropna()
        fig = go.Figure(go.Bar(x=m["K"], y=m["Silhouette"], text=m["Silhouette"].round(3), textposition="outside",
                               marker_color=[GOLD if k == K else "#D6D2C8" for k in m["K"]]))
        fig.update_layout(title="Silhouette score by K", xaxis=dict(dtick=1, title="K"),
                          yaxis=dict(title="Silhouette", range=[0, m["Silhouette"].max() * 1.18]))
        st.plotly_chart(style(fig, 380, legend=False), width="stretch")
        html('<p class="note">The Silhouette score ranges from -1 to 1 and measures how well separated the '
             f'clusters are. It peaks at K = {int(m.loc[m.Silhouette.idxmax(), "K"])}.</p>')

    html('<div class="section-label">Configuration</div><div class="section-title">Final model</div>')
    st.dataframe(
        pd.DataFrame({
            "Parameter": ["Algorithm", "Features", "Scaling", "Number of clusters", "Initialisations", "Random state",
                          "Silhouette score"],
            "Value": ["K-Means (scikit-learn)", "Annual Income, Spending Score", "StandardScaler", str(K), "10", "42",
                      f"{sil:.3f}"],
        }),
        width="stretch", hide_index=True,
    )

# ----------------------------------------------------------------------------
# Segments
# ----------------------------------------------------------------------------
with tabs[3]:
    html(
        f"""
        <div class="section-label">Findings</div><div class="section-title">Five customer segments</div>
        <p class="section-sub">Each segment is named from its average income and spending score, with a
        recommended marketing approach.</p>
        """
    )
    share = "".join(
        f'<div title="{r.Segment}: {int(r.Customers)}" style="width:{r.Customers / len(df) * 100}%;'
        f'background:{cmap[r.Segment]}"></div>'
        for r in summary.itertuples()
    )
    rows = "".join(
        f"""
        <div class="seg-row">
          <div class="seg-name"><span class="swatch" style="background:{cmap[r.Segment]}"></span>
            <div><div class="t">{r.Segment}</div><div class="s">Cluster {c} &middot; {SEGMENTS[labels[c]]['rule']}</div></div></div>
          <div class="num">{int(r.Customers)}<small>{r.Customers / len(df):.0%} of total</small></div>
          <div class="num">${r.Avg_Income:.0f}k<small>avg. income</small></div>
          <div class="num">{r.Avg_Spending:.0f}<small>avg. score</small></div>
          <div class="num">{r.Avg_Age:.0f}<small>avg. age</small></div>
          <div class="strategy">{SEGMENTS[labels[c]]['strategy']}</div>
        </div>
        """
        for c, r in zip(summary.index, summary.itertuples())
    )
    html(
        f"""
        <div class="share">{share}</div>
        <div class="seg-table">
          <div class="seg-row head"><div>Segment</div><div>Customers</div><div>Income</div><div>Spending</div>
          <div>Age</div><div>Recommended strategy</div></div>
          {rows}
        </div>
        """
    )

    html('<div class="section-label">Visualisation</div><div class="section-title">Segments in feature space</div>')
    view = st.segmented_control("View", ["Income vs spending", "Age, income and spending (3D)"],
                                default="Income vs spending", label_visibility="collapsed")
    if view == "Age, income and spending (3D)":
        fig = px.scatter_3d(df, x="Age", y=FEATURES[0], z=FEATURES[1], color="Segment", color_discrete_map=cmap,
                            category_orders={"Segment": seg_order})
        fig.update_traces(marker=dict(size=4, line=dict(width=0)))
    else:
        fig = px.scatter(df, x=FEATURES[0], y=FEATURES[1], color="Segment", color_discrete_map=cmap,
                         category_orders={"Segment": seg_order}, hover_data=["CustomerID", "Gender", "Age"])
        fig.update_traces(marker=dict(size=10, opacity=0.9, line=dict(width=0.8, color="white")))
    st.plotly_chart(style(fig, 520), width="stretch")

    html('<div class="section-label">Records</div><div class="section-title">Browse customers</div>')
    pick = st.selectbox("Segment", seg_order)
    st.dataframe(df[df["Segment"] == pick], width="stretch", hide_index=True, height=300)
    st.download_button("Download segmented customers (CSV)", df.to_csv(index=False), "segmented_customers.csv",
                       "text/csv")

# ----------------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------------
with tabs[4]:
    html(
        """
        <div class="section-label">Live model</div><div class="section-title">Classify a new customer</div>
        <p class="section-sub">Enter a customer's profile. The trained model assigns them to the segment with the
        nearest centroid.</p>
        """
    )
    left, right = st.columns([2, 3], gap="large")
    with left:
        with st.container(border=True):
            name = st.text_input("Customer name", "New customer")
            g1, g2 = st.columns(2)
            gender = g1.selectbox("Gender", ["Female", "Male"])
            age = g2.number_input("Age", 18, 80, 30)
            income = st.slider("Annual income (k$)", 10, 150, 70)
            spending = st.slider("Spending score (1-100)", 1, 100, 80)
        html('<p class="note">Gender and age are recorded for context. As in the notebook, the model uses '
             'annual income and spending score.</p>')

    x_new = scaler.transform(pd.DataFrame([[income, spending]], columns=FEATURES))
    cluster = int(model.predict(x_new)[0])
    dists = np.linalg.norm(model.cluster_centers_ - x_new, axis=1)
    seg = SEGMENTS[labels[cluster]]
    peers = summary.loc[cluster]

    with right:
        html(
            f"""
            <div class="result">
              <div class="eyebrow">Predicted segment</div>
              <div class="name">{seg['name']}</div>
              <div class="rule">Cluster {cluster} &middot; {seg['rule']}</div>
              <hr>
              <div class="k">Recommended strategy</div>
              <p>{seg['strategy']}</p>
              <div class="facts">
                <div><div class="k">Segment size</div><div class="v">{int(peers.Customers)}</div></div>
                <div><div class="k">Peer avg. income</div><div class="v">${peers.Avg_Income:.0f}k</div></div>
                <div><div class="k">Peer avg. score</div><div class="v">{peers.Avg_Spending:.0f}</div></div>
              </div>
            </div>
            """
        )
        fig = px.scatter(df, x=FEATURES[0], y=FEATURES[1], color="Segment", color_discrete_map=cmap,
                         category_orders={"Segment": seg_order})
        fig.update_traces(marker=dict(size=7, opacity=0.35))
        fig.add_trace(go.Scatter(x=[income], y=[spending], mode="markers+text", name=name, text=[name],
                                 textposition="top center", textfont=dict(color=INK),
                                 marker=dict(symbol="diamond", size=16, color=seg["color"], line=dict(width=2, color=INK))))
        st.plotly_chart(style(fig, 360), width="stretch")

    with st.expander("Distance to each centroid"):
        st.dataframe(
            pd.DataFrame({
                "Segment": [SEGMENTS[labels[c]]["name"] for c in range(K)],
                "Cluster": range(K),
                "Distance (scaled units)": dists.round(3),
            }).sort_values("Distance (scaled units)"),
            width="stretch", hide_index=True,
        )

# ----------------------------------------------------------------------------
# Batch
# ----------------------------------------------------------------------------
with tabs[5]:
    html(
        f"""
        <div class="section-label">Bulk scoring</div><div class="section-title">Segment a customer list</div>
        <p class="section-sub">Upload a CSV containing the columns <b>{FEATURES[0]}</b> and <b>{FEATURES[1]}</b>.
        Every row is assigned a segment and a recommended strategy.</p>
        """
    )
    sample = df_raw.sample(10, random_state=1)[["CustomerID", "Gender", "Age"] + FEATURES]
    st.download_button("Download a sample CSV", sample.to_csv(index=False), "sample_customers.csv", "text/csv")

    up = st.file_uploader("Upload customers CSV", type="csv")
    if up is not None:
        new = pd.read_csv(up)
        missing = [f for f in FEATURES if f not in new.columns]
        if missing:
            st.error(f"Missing column(s): {', '.join(missing)}")
        else:
            new = new.dropna(subset=FEATURES)
            pred = model.predict(scaler.transform(new[FEATURES]))
            new["Cluster"] = pred
            new["Segment"] = [SEGMENTS[labels[c]]["name"] for c in pred]
            new["Strategy"] = [SEGMENTS[labels[c]]["strategy"] for c in pred]
            html(f'<p class="note" style="margin-top:1rem">{len(new)} customers segmented.</p>')
            c1, c2 = st.columns([3, 2], gap="large")
            with c1:
                st.dataframe(new, width="stretch", hide_index=True)
            with c2:
                fig = px.pie(new, names="Segment", hole=0.62, color="Segment", color_discrete_map=cmap)
                fig.update_traces(textinfo="percent", marker=dict(line=dict(color="#FAF9F6", width=2)))
                st.plotly_chart(style(fig, 340), width="stretch")
            st.download_button("Download results (CSV)", new.to_csv(index=False), "predicted_segments.csv", "text/csv")

html(
    f"""
    <div class="foot">
      <div>E-Commerce Customer Segmentation &middot; K-Means Clustering</div>
      <div>Python &middot; Pandas &middot; Scikit-learn &middot; Plotly &middot; Streamlit</div>
    </div>
    """
)
