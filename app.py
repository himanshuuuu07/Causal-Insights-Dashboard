import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from utils.data_processor import validate_and_clean, generate_sample_data
from utils.causal_analysis import discover_causal_structure, generate_causal_graph, detect_metric_change
from utils.ai_summary import generate_insight_summary, answer_whatif_query

st.set_page_config(page_title="Causal Insights Dashboard", layout="wide", page_icon="🔍")

st.markdown("""
<style> 
    .main-header {font-size: 42px; font-weight: 700; color: #FF4B4B; margin-bottom: 10px;}
    .sub-header {font-size: 18px; color: #666; margin-bottom: 30px;}
    .insight-box {background: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #FF4B4B;}
    .metric-card {background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🔍 Causal Insights Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automatically discover why your metrics changed — built for Fire AI</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📊 Data Input")
    data_source = st.radio("Choose data source:", ["Upload CSV", "Use Demo Dataset"])
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your business data", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df = validate_and_clean(df)
            st.success(f"✅ Dataset loaded ({len(df)} rows)")
    else:
        df = generate_sample_data()
        st.success("✅ Demo retail dataset loaded (294 days)")

    st.markdown("---")
    st.markdown("### ⚙️ Analysis Settings")

    if 'df' in locals():
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        target_metric = st.selectbox(
            "Select target metric:",
            numeric_cols,
            index=numeric_cols.index('Revenue') if 'Revenue' in numeric_cols else 0
        )
        window_days = st.slider("Comparison window (days):", 7, 90, 30)
        top_n_features = st.slider("Top causal factors to show:", 3, 10, 5)

if 'df' in locals() and target_metric:
    if st.button("🔬 Find Root Cause", type="primary", use_container_width=True):
        with st.spinner("Analyzing causal relationships..."):
            metric_change = detect_metric_change(df, target_metric, window=window_days)
            importance_df, model, feature_cols = discover_causal_structure(df, target_metric)
            G = generate_causal_graph(importance_df, top_n=top_n_features)
            summary = generate_insight_summary(df, target_metric, importance_df, metric_change, top_n=3)

            st.session_state.analysis_done = True
            st.session_state.metric_change = metric_change
            st.session_state.importance_df = importance_df
            st.session_state.G = G
            st.session_state.summary = summary
            st.session_state.model = model
            st.session_state.feature_cols = feature_cols
            st.session_state.target_metric = target_metric
            st.session_state.window_days = window_days
            st.session_state.top_n_features = top_n_features

    if st.session_state.get('analysis_done'):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label=f"Current {st.session_state.target_metric} (Last {st.session_state.window_days}d)",
                value=f"₹{st.session_state.metric_change['recent_avg']:,.0f}",
                delta=f"{st.session_state.metric_change['pct_change']:.1f}%"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label=f"Previous {st.session_state.target_metric}",
                value=f"₹{st.session_state.metric_change['previous_avg']:,.0f}"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            top_driver = st.session_state.importance_df.iloc[0]
            st.metric(
                label="Top Causal Driver",
                value=top_driver['Feature'],
                delta=f"Importance: {top_driver['Importance']:.3f}"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### 🧠 Causal Chain Explanation")
        st.markdown(f'<div class="insight-box">{st.session_state.summary}</div>', unsafe_allow_html=True)

        st.markdown("---")

        col_graph, col_features = st.columns([1.2, 1])

        with col_graph:
            st.markdown("### 🕸️ Causal Graph")
            pos = nx.spring_layout(st.session_state.G, k=0.5, iterations=50)

            edge_trace = go.Scatter(x=[], y=[], line=dict(width=2, color='#888'), hoverinfo='none', mode='lines')
            for edge in st.session_state.G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += tuple([x0, x1, None])
                edge_trace['y'] += tuple([y0, y1, None])

            node_trace = go.Scatter(
                x=[], y=[], text=[], mode='markers+text', hoverinfo='text',
                marker=dict(size=[], color=[], line=dict(width=2)),
                textposition="top center"
            )
            for node in st.session_state.G.nodes():
                x, y = pos[node]
                node_trace['x'] += tuple([x])
                node_trace['y'] += tuple([y])
                node_trace['text'] += tuple([node])
                node_trace['marker']['size'] += tuple([50 if node == st.session_state.target_metric else 30])
                node_trace['marker']['color'] += tuple(['#FF4B4B' if node == st.session_state.target_metric else '#00C9FF'])

            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=400
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_features:
            st.markdown("### 📊 Feature Importance")
            top_features = st.session_state.importance_df.head(st.session_state.top_n_features)
            fig_bar = go.Figure(go.Bar(
                x=top_features['Importance'],
                y=top_features['Feature'],
                orientation='h',
                marker=dict(color='#FF4B4B')
            ))
            fig_bar.update_layout(
                xaxis_title="Causal Importance",
                yaxis_title="",
                height=400,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        st.markdown("### 💬 Ask What-If Questions")
        user_question = st.text_input("Example: 'What if Ad_Spend increases by 20%?'")
        if user_question:
            with st.spinner("Simulating scenario..."):
                answer = answer_whatif_query(
                    user_question,
                    df,
                    st.session_state.model,
                    st.session_state.feature_cols
                )
                st.success(answer)
else:
    st.info("👈 Upload data or select demo dataset to start analysis")

st.markdown("---")
st.markdown("Built for **Fire AI** | Developed by Himanshu Maurya | [LinkedIn](https://linkedin.com/in/himanshhh07/)")
