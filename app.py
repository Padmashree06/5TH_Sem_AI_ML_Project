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

# Segment names and marketing strategies, based on the same income/spending
# rules used in the notebook's "Customer Segment Analysis" cell.
SEGMENTS = {
    "high_high": {
        "name": "Premium Shoppers",
        "rule": "High income, high spending",
        "icon": "💎",
        "color": "#4F46E5",
        "strategy": "VIP loyalty programme, early access to new launches, premium bundles and personalised recommendations.",
    },
    "high_low": {
        "name": "Careful Affluent",
        "rule": "High income, low spending",
        "icon": "🧐",
        "color": "#0EA5E9",
        "strategy": "Win them over with quality-focused messaging, product reviews, free trials and targeted personalised offers.",
    },
    "low_high": {
        "name": "Impulsive Spenders",
        "rule": "Lower income, high spending",
        "icon": "🛍️",
        "color": "#EC4899",
        "strategy": "Flash sales, 'Buy Now Pay Later', trending products and limited-time deals pushed via app notifications.",
    },
    "low_low": {
        "name": "Budget Conscious",
        "rule": "Lower income, low spending",
        "icon": "💰",
        "color": "#F59E0B",
        "strategy": "Discount coupons, value packs, free-shipping thresholds and price-drop alerts.",
    },
    "mid": {
        "name": "Standard Customers",
        "rule": "Moderate income and spending",
        "icon": "🙂",
        "color": "#10B981",
        "strategy": "Regular newsletters, cross-selling, seasonal campaigns and reward points to nudge them upward.",
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
def train_model(df: pd.DataFrame, k: int):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    return scaler, model, X_scaled


def label_clusters(df: pd.DataFrame) -> dict:
    """Map each cluster id to a segment name using its average income/spending."""
    labels = {}
    for c, g in df.groupby("Cluster"):
        key = segment_key(g[FEATURES[0]].mean(), g[FEATURES[1]].mean())
        labels[c] = key
    return labels


def cluster_display_name(c: int, labels: dict) -> str:
    seg = SEGMENTS[labels[c]]
    return f"C{c} · {seg['name']}"


def color_map(labels: dict) -> dict:
    return {cluster_display_name(c, labels): SEGMENTS[key]["color"] for c, key in labels.items()}


# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛒",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      .hero {
        background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #EC4899 100%);
        color: white; padding: 2rem 2.2rem; border-radius: 18px; margin-bottom: 1.4rem;
      }
      .hero h1 {color: white; margin: 0 0 .4rem 0; font-size: 2.1rem;}
      .hero p {margin: 0; opacity: .92; font-size: 1.05rem;}
      .seg-card {
        border-radius: 14px; padding: 1rem 1.1rem; height: 100%;
        background: #F8F9FE; border-left: 6px solid var(--c);
      }
      .seg-card h4 {margin: 0 0 .3rem 0; font-size: 1.05rem; padding: 0;}
      .seg-card .muted {color: #6B7280; font-size: .88rem;}
      .result-box {
        border-radius: 16px; padding: 1.4rem 1.6rem; color: white;
        background: var(--c);
      }
      .result-box h2 {color: white; margin: 0;}
      div[data-testid="stMetric"] {
        background: #F4F5FB; border-radius: 12px; padding: .8rem 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

df_raw = load_data()

# Sidebar
with st.sidebar:
    st.markdown("## 🛒 Customer Segmentation")
    st.caption("K-Means clustering on the Mall Customers dataset")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Data Exploration", "📈 Choosing K", "🎯 Customer Segments", "🔮 Predict Segment", "📂 Batch Prediction"],
        label_visibility="collapsed",
    )
    st.divider()
    k = st.slider("Number of clusters (K)", 2, 10, 5, help="The notebook uses K = 5, chosen with the Elbow Method.")
    st.divider()
    st.markdown(
        "**Tech stack**  \nPython · Pandas · Scikit-learn · Plotly · Streamlit\n\n"
        "[Dataset (Kaggle)](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)  \n"
        "[GitHub repo](https://github.com/Padmashree06/5TH_Sem_AI_ML_Project)"
    )

scaler, model, X_scaled = train_model(df_raw, k)
df = df_raw.copy()
df["Cluster"] = model.labels_
labels = label_clusters(df)
df["Segment"] = df["Cluster"].map(lambda c: cluster_display_name(c, labels))
cmap = color_map(labels)
sil = silhouette_score(X_scaled, model.labels_)

# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
if page == "🏠 Overview":
    st.markdown(
        """
        <div class="hero">
          <h1>E-Commerce Customer Segmentation</h1>
          <p>Grouping customers by <b>annual income</b> and <b>spending score</b> using
          unsupervised K-Means clustering, so a store can target each group with the right marketing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", len(df))
    c2.metric("Features used", len(FEATURES))
    c3.metric("Clusters (K)", k)
    c4.metric("Silhouette score", f"{sil:.3f}")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Segmentation at a glance")
        fig = px.scatter(
            df, x=FEATURES[0], y=FEATURES[1], color="Segment",
            color_discrete_map=cmap, hover_data=["CustomerID", "Gender", "Age"],
            category_orders={"Segment": sorted(df["Segment"].unique())},
        )
        centers = scaler.inverse_transform(model.cluster_centers_)
        fig.add_trace(go.Scatter(
            x=centers[:, 0], y=centers[:, 1], mode="markers", name="Centroids",
            marker=dict(symbol="x", size=16, color="black", line=dict(width=2)),
        ))
        fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="white")), selector=dict(mode="markers", type="scatter"))
        fig.update_layout(height=460, legend=dict(orientation="h", y=-0.2), margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Project pipeline")
        st.markdown(
            """
            1. **Load data** – 200 customers, 5 attributes
            2. **Explore (EDA)** – age, gender, income & spending distributions
            3. **Preprocess** – select income & spending, apply `StandardScaler`
            4. **Choose K** – Elbow Method (WCSS) + Silhouette score
            5. **Train K-Means** – `n_clusters=5`, `random_state=42`
            6. **Evaluate** – Silhouette score
            7. **Interpret** – name each segment & suggest a marketing strategy
            8. **Deploy** – this interactive app 🎉
            """
        )
        st.info("Use the sidebar to explore the data, see how K was chosen, and predict the segment of a new customer.")

elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration")

    tab1, tab2, tab3 = st.tabs(["Dataset", "Distributions", "Relationships"])
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df_raw.shape[0])
        c2.metric("Columns", df_raw.shape[1])
        c3.metric("Missing values", int(df_raw.isnull().sum().sum()))
        c4.metric("Female / Male", f"{(df_raw['Gender'] == 'Female').sum()} / {(df_raw['Gender'] == 'Male').sum()}")
        st.dataframe(df_raw, use_container_width=True, height=320, hide_index=True)
        st.subheader("Statistical summary")
        st.dataframe(df_raw.drop(columns="CustomerID").describe().round(2), use_container_width=True)

    with tab2:
        col = st.selectbox("Choose a feature", ["Age", FEATURES[0], FEATURES[1]])
        split = st.toggle("Split by gender", value=False)
        fig = px.histogram(
            df_raw, x=col, nbins=15, color="Gender" if split else None, marginal="box",
            barmode="overlay", opacity=0.75, color_discrete_sequence=["#4F46E5", "#EC4899"],
        )
        fig.update_layout(height=430, bargap=0.05, yaxis_title="Number of Customers")
        st.plotly_chart(fig, use_container_width=True)

        g1, g2 = st.columns(2)
        with g1:
            fig = px.pie(df_raw, names="Gender", hole=0.5, title="Gender split",
                         color_discrete_sequence=["#EC4899", "#4F46E5"])
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            fig = px.box(df_raw, x="Gender", y=FEATURES[1], color="Gender", title="Spending score by gender",
                         color_discrete_sequence=["#4F46E5", "#EC4899"])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.scatter(df_raw, x=FEATURES[0], y=FEATURES[1], color="Gender", hover_data=["Age"],
                         title="Annual Income vs Spending Score", color_discrete_sequence=["#4F46E5", "#EC4899"])
        fig.update_traces(marker=dict(size=9))
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Five visible groups in this plot are the motivation for choosing K = 5.")

        corr = df_raw.drop(columns=["CustomerID", "Gender"]).corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation heatmap")
        st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Choosing K":
    st.title("📈 Choosing the Number of Clusters")
    metrics = elbow_and_silhouette(X_scaled)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = px.line(metrics, x="K", y="WCSS", markers=True, title="Elbow Method (WCSS)")
        fig.add_vline(x=5, line_dash="dash", line_color="#EC4899", annotation_text="Elbow at K = 5")
        fig.update_layout(height=420, xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**WCSS** (within-cluster sum of squares) keeps falling as K grows. "
                    "The *elbow* — where the drop flattens out — suggests a good K.")
    with c2:
        m = metrics.dropna()
        fig = px.bar(m, x="K", y="Silhouette", title="Silhouette score by K", text_auto=".3f",
                     color=m["K"] == 5, color_discrete_map={True: "#EC4899", False: "#A5B4FC"})
        fig.update_layout(height=420, showlegend=False, xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)
        best = int(m.loc[m["Silhouette"].idxmax(), "K"])
        st.markdown(f"**Silhouette score** (−1 to 1) measures how well separated clusters are. "
                    f"Highest here at **K = {best}**.")

    st.dataframe(metrics.round(3), use_container_width=True, hide_index=True)

elif page == "🎯 Customer Segments":
    st.title("🎯 Customer Segments")
    st.caption(f"K-Means with K = {k} · Silhouette score = {sil:.3f}")

    summary = df.groupby("Segment").agg(
        Customers=("CustomerID", "count"),
        Avg_Age=("Age", "mean"),
        Avg_Income=(FEATURES[0], "mean"),
        Avg_Spending=(FEATURES[1], "mean"),
        Female_pct=("Gender", lambda s: (s == "Female").mean() * 100),
    ).round(1).sort_index()

    cols = st.columns(min(len(summary), 5))
    for i, (seg_name, row) in enumerate(summary.iterrows()):
        c = int(seg_name.split(" ")[0][1:])
        seg = SEGMENTS[labels[c]]
        with cols[i % len(cols)]:
            st.markdown(
                f"""
                <div class="seg-card" style="--c:{seg['color']}">
                  <h4>{seg['icon']} {seg['name']}</h4>
                  <div class="muted">Cluster {c} · {seg['rule']}</div>
                  <p style="margin:.6rem 0 .2rem 0"><b>{int(row.Customers)}</b> customers</p>
                  <div class="muted">Income ${row.Avg_Income:.0f}k · Score {row.Avg_Spending:.0f} · Age {row.Avg_Age:.0f}</div>
                  <p style="margin-top:.6rem;font-size:.9rem">🎯 {seg['strategy']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.write("")

    view = st.radio("View", ["2D: Income vs Spending", "3D: Age, Income & Spending"], horizontal=True)
    if view.startswith("2D"):
        fig = px.scatter(df, x=FEATURES[0], y=FEATURES[1], color="Segment", color_discrete_map=cmap,
                         hover_data=["CustomerID", "Gender", "Age"], category_orders={"Segment": list(summary.index)})
        fig.update_traces(marker=dict(size=11, line=dict(width=0.6, color="white")))
    else:
        fig = px.scatter_3d(df, x="Age", y=FEATURES[0], z=FEATURES[1], color="Segment", color_discrete_map=cmap,
                            category_orders={"Segment": list(summary.index)})
        fig.update_traces(marker=dict(size=5))
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Cluster summary")
        st.dataframe(summary.rename(columns={"Female_pct": "Female %"}), use_container_width=True)
    with c2:
        fig = px.bar(summary.reset_index(), x="Segment", y="Customers", color="Segment",
                     color_discrete_map=cmap, title="Customers per segment", text_auto=True)
        fig.update_layout(showlegend=False, height=330, xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    seg_pick = st.selectbox("Browse customers in a segment", list(summary.index))
    st.dataframe(df[df["Segment"] == seg_pick].drop(columns="Segment"), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download segmented customers (CSV)", df.to_csv(index=False), "segmented_customers.csv", "text/csv")

elif page == "🔮 Predict Segment":
    st.title("🔮 Predict a New Customer's Segment")
    st.caption("Enter a customer's details – the trained K-Means model assigns them to the nearest cluster centroid.")

    left, right = st.columns([2, 3], gap="large")
    with left:
        name = st.text_input("Customer name (optional)", "New Customer")
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        age = st.slider("Age", 18, 70, 30)
        income = st.slider("Annual Income (k$)", 10, 150, 70)
        spending = st.slider("Spending Score (1-100)", 1, 100, 80)
        st.caption("Gender and age are shown for context; the model uses income and spending score (as in the notebook).")

    x_new = scaler.transform(pd.DataFrame([[income, spending]], columns=FEATURES))
    cluster = int(model.predict(x_new)[0])
    dists = np.linalg.norm(model.cluster_centers_ - x_new, axis=1)
    seg = SEGMENTS[labels[cluster]]

    with right:
        st.markdown(
            f"""
            <div class="result-box" style="--c:{seg['color']}">
              <div style="opacity:.85">{name} belongs to</div>
              <h2>{seg['icon']} {seg['name']}</h2>
              <div style="opacity:.9">Cluster {cluster} · {seg['rule']}</div>
              <p style="margin:.8rem 0 0 0">🎯 <b>Recommended strategy:</b> {seg['strategy']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig = px.scatter(df, x=FEATURES[0], y=FEATURES[1], color="Segment", color_discrete_map=cmap, opacity=0.45,
                         category_orders={"Segment": sorted(df["Segment"].unique())})
        fig.add_trace(go.Scatter(
            x=[income], y=[spending], mode="markers+text", name=name, text=["  ★ " + name], textposition="middle right",
            marker=dict(symbol="star", size=24, color=seg["color"], line=dict(width=2, color="black")),
        ))
        fig.update_layout(height=400, margin=dict(t=20), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("How was this decided? (distance to each centroid)"):
        d = pd.DataFrame({
            "Segment": [cluster_display_name(c, labels) for c in range(k)],
            "Distance (scaled)": dists.round(3),
        }).sort_values("Distance (scaled)")
        st.dataframe(d, use_container_width=True, hide_index=True)
        st.caption("K-Means assigns a point to the cluster with the closest centroid (smallest distance).")

elif page == "📂 Batch Prediction":
    st.title("📂 Batch Prediction")
    st.markdown(f"Upload a CSV with the columns **`{FEATURES[0]}`** and **`{FEATURES[1]}`** to segment many customers at once.")

    sample = df_raw.sample(10, random_state=1)[["CustomerID", "Gender", "Age"] + FEATURES]
    st.download_button("⬇️ Download a sample CSV", sample.to_csv(index=False), "sample_customers.csv", "text/csv")

    up = st.file_uploader("Upload customers CSV", type="csv")
    if up is not None:
        new = pd.read_csv(up)
        missing = [f for f in FEATURES if f not in new.columns]
        if missing:
            st.error(f"Missing column(s): {missing}")
        else:
            new = new.dropna(subset=FEATURES)
            pred = model.predict(scaler.transform(new[FEATURES]))
            new["Cluster"] = pred
            new["Segment"] = [SEGMENTS[labels[c]]["name"] for c in pred]
            new["Strategy"] = [SEGMENTS[labels[c]]["strategy"] for c in pred]
            st.success(f"Segmented {len(new)} customers.")
            st.dataframe(new, use_container_width=True, hide_index=True)
            fig = px.pie(new, names="Segment", hole=0.5, title="Segment mix of uploaded customers",
                         color="Segment", color_discrete_map={s["name"]: s["color"] for s in SEGMENTS.values()})
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇️ Download results", new.to_csv(index=False), "predicted_segments.csv", "text/csv")
