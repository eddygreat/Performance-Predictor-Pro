import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from streamlit_shap import st_shap
import shap

def apply_custom_css():
    st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stButton>button {
            background: linear-gradient(45deg, #2e7d32, #4caf50);
            color: white; border-radius: 12px;
            padding: 0.6rem 2.5rem; font-weight: 700;
            border: none; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }
        .metric-card {
            background-color: white;
            padding: 1.5rem; border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.05);
            border-left: 5px solid #2e7d32;
        }
        .stMetric { background-color: #ffffff; border-radius: 10px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    st.title("🎓 Student Success Dashboard")
    st.markdown("---")

def render_shap_plot(shap_values, X_input, instance_index=0):
    """Renders a SHAP waterfall plot for a specific prediction."""
    st.subheader("🔍 Prediction Breakdown (Explainable AI)")
    st.write("This chart shows how each factor contributed to the result.")
    # st_shap(shap.plots.waterfall(shap_values[instance_index], max_display=10))
    # Note: Waterfall expects a single Explainer result row
    st_shap(shap.plots.bar(shap_values[instance_index], max_display=10))

def render_batch_charts(results_df):
    """Renders advanced Plotly charts for batch predictions."""
    st.subheader("📊 Batch Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pass/Fail Pie Chart
        fig_pie = px.pie(results_df, names='Prediction', 
                         title='Pass vs. Fail Distribution',
                         color='Prediction',
                         color_discrete_map={'PASS': '#2e7d32', 'FAIL': '#d32f2f'},
                         hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        # Confidence Distribution
        results_df['ConfNum'] = results_df['Confidence'].str.strip('%').astype(float)
        fig_hist = px.histogram(results_df, x='ConfNum', 
                                title='Model Confidence Distribution',
                                labels={'ConfNum': 'Confidence (%)'},
                                color_discrete_sequence=['#1976d2'])
        st.plotly_chart(fig_hist, use_container_width=True)
        
    # Correlation with key factors
    st.subheader("📈 Factor Analysis")
    # For simplicity, we just look at top 3 features vs prediction
    # If G3 isn't available, we use the prediction as target
    fig_scatter = px.box(results_df, x='Prediction', y='absences', 
                         color='Prediction', points="all",
                         title='Absences vs. Predicted Performance',
                         color_discrete_map={'PASS': '#2e7d32', 'FAIL': '#d32f2f'})
    st.plotly_chart(fig_scatter, use_container_width=True)

def display_metrics(results_df):
    total = len(results_df)
    passes = len(results_df[results_df['Prediction'] == 'PASS'])
    fails = total - passes
    pass_rate = (passes / total) * 100 if total > 0 else 0
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Students", total)
    m2.metric("Predicted Passes", passes)
    m3.metric("Predicted Fails", fails, delta=f"{(fails/total)*100:.1f}%", delta_color="inverse")
    m4.metric("Success Rate", f"{pass_rate:.1f}%")
