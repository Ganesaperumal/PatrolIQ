import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="PatrolIQ - Chicago Crime Analysis", layout="wide", page_icon="🚓")

@st.cache_data
def load_data():
    # Load the clustered 30k sample for fast dashboard interaction
    data_path = os.path.join(os.path.dirname(__file__), '../data/processed/clustered_crimes_sample.csv')
    return pd.read_csv(data_path)

df = load_data()

# CSS Styling to hit the Dynamic/Rich Aesthetic requirements
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main-title {
        font-size: 3rem;
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        margin-bottom: 30px;
    }
    [data-testid="stSidebar"] {
        background-color: #1e222d;
    }
    .stPlotlyChart {
         border-radius: 10px;
         box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.4);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">PatrolIQ Smart Intelligence Platform</div>', unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Overview & Temporal Metrics", "Geographic Hotspots", "Dimensionality Reduction"])
st.sidebar.markdown("---")
st.sidebar.info("Designed for Chicago PD Resource Optimization.")

# 1. Overview Page
if page == "Overview & Temporal Metrics":
    st.header("Temporal Crime Patterns")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Crime Volume by Hour")
        hourly_crime = df.groupby('Hour').size().reset_index(name='Counts')
        fig = px.bar(hourly_crime, x='Hour', y='Counts', color='Counts', color_continuous_scale='Inferno')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Severity Distribution")
        # Ensure Crime_Severity_Score is categorized
        sev_counts = df['Crime_Severity_Score'].value_counts().reset_index()
        sev_counts.columns = ['Severity', 'Count']
        fig2 = px.pie(sev_counts, values='Count', names='Severity', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig2, use_container_width=True)
        
    st.markdown("### Temporal Clustering Analysis")
    st.write("Using K-Means to identify distinct time-based risk periods. Below represents cyclical peaks.")
    fig_temp = px.scatter(df, x='Hour_Cos', y='Hour_Sin', color='Temp_Cluster_KMeans', title="Cyclical Time Clustering", hover_data=['Hour'])
    st.plotly_chart(fig_temp, use_container_width=True)
    
# 2. Geographic Hotspots Page
elif page == "Geographic Hotspots":
    st.header("Chicago Hotspot Heatmaps")
    cluster_alg = st.selectbox("Select Clustering Algorithm", ["Geo_Cluster_KMeans", "Geo_Cluster_DBSCAN", "Geo_Cluster_Agg"])
    
    # Render map
    fig_map = px.scatter_mapbox(df, 
                                lat="Latitude", 
                                lon="Longitude", 
                                color=cluster_alg, 
                                hover_name="Primary Type", 
                                zoom=9, 
                                height=600,
                                mapbox_style="carto-darkmatter",
                                color_continuous_scale=px.colors.cyclical.IceFire)
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

# 3. PCA & tSNE Page
elif page == "Dimensionality Reduction":
    st.header("Crime Typology - Advanced Dimensionality Reduction")
    st.markdown("We compressed 22+ variables into 2/3 Dimensional space to separate different behavioral crime patterns.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("PCA Representation (3 Components)")
        fig_pca = px.scatter_3d(df, x='PCA_1', y='PCA_2', z='PCA_3', color='Crime_Severity_Score', size_max=10, opacity=0.7)
        fig_pca.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_pca, use_container_width=True)
        
    with col_b:
        st.subheader("t-SNE Representation (2D)")
        fig_tsne = px.scatter(df, x='tSNE_1', y='tSNE_2', color='Geo_Cluster_KMeans', opacity=0.6)
        st.plotly_chart(fig_tsne, use_container_width=True)
