"""
Interactive dashboard for exploring stellar anomalies.
Built with Streamlit for easy deployment and interaction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys

# Add src to path
sys.path.append('../src')
from visualization import StellarVisualization

# Page configuration
st.set_page_config(
    page_title="Astrophysical Oddball Finder",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load processed data and results."""
    try:
        # Load anomaly results
        results_dir = "../results"
        
        # Try to load different anomaly result files
        anomaly_files = {
            'combined': 'top_anomalies_combined.csv',
            'isolation_forest': 'top_anomalies_isolation_forest.csv',
            'autoencoder': 'top_anomalies_autoencoder.csv',
            'dbscan': 'top_anomalies_dbscan.csv'
        }
        
        results = {}
        for method, filename in anomaly_files.items():
            filepath = os.path.join(results_dir, filename)
            if os.path.exists(filepath):
                results[method] = pd.read_csv(filepath)
        
        # Load original processed data if available
        processed_data = None
        processed_file = "../data/gaia_dr3_processed.csv"
        if os.path.exists(processed_file):
            processed_data = pd.read_csv(processed_file)
        
        return results, processed_data
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {}, None

@st.cache_data
def load_anomaly_scores():
    """Load anomaly scores if available."""
    try:
        scores_file = "../results/anomaly_scores.joblib"
        if os.path.exists(scores_file):
            return joblib.load(scores_file)
        return None
    except Exception as e:
        st.error(f"Error loading anomaly scores: {e}")
        return None

def create_hr_diagram(df, anomaly_data=None, highlight_source=None):
    """Create interactive HR diagram."""
    
    # Filter for valid data
    valid_mask = (
        df['bp_rp'].notna() & df['abs_g_mag'].notna() &
        (df['bp_rp'] > -1) & (df['bp_rp'] < 4) &
        (df['abs_g_mag'] > -5) & (df['abs_g_mag'] < 20)
    )
    df_plot = df[valid_mask].copy()
    
    # Create base scatter plot
    fig = px.scatter(
        df_plot.sample(min(50000, len(df_plot))),  # Sample for performance
        x='bp_rp', 
        y='abs_g_mag',
        opacity=0.3,
        color_discrete_sequence=['lightblue'],
        labels={
            'bp_rp': 'BP - RP Color (mag)',
            'abs_g_mag': 'Absolute G Magnitude'
        },
        title='Hertzsprung-Russell Diagram with Anomalies'
    )
    
    # Add anomalies if available
    if anomaly_data is not None and len(anomaly_data) > 0:
        fig.add_scatter(
            x=anomaly_data['bp_rp'],
            y=anomaly_data['abs_g_mag'],
            mode='markers',
            marker=dict(
                size=6,
                color=anomaly_data['anomaly_score'],
                colorscale='Viridis',
                colorbar=dict(title="Anomaly Score"),
                symbol='star',
                line=dict(width=1, color='black')
            ),
            name='Anomalies',
            hovertemplate='<b>Anomaly</b><br>' +
                         'Source ID: %{customdata[0]}<br>' +
                         'BP-RP: %{x:.3f}<br>' +
                         'Abs G Mag: %{y:.3f}<br>' +
                         'Anomaly Score: %{customdata[1]:.3f}',
            customdata=np.column_stack([
                anomaly_data['source_id'].values,
                anomaly_data['anomaly_score'].values
            ])
        )
    
    # Highlight specific source if requested
    if highlight_source is not None:
        source_data = anomaly_data[anomaly_data['source_id'] == highlight_source]
        if len(source_data) > 0:
            fig.add_scatter(
                x=source_data['bp_rp'],
                y=source_data['abs_g_mag'],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star',
                    line=dict(width=3, color='yellow')
                ),
                name='Selected Source',
                showlegend=True
            )
    
    # Add main sequence line
    color_ms = np.linspace(-0.5, 2.0, 100)
    mag_ms = 4.83 + 5.5 * color_ms
    fig.add_scatter(
        x=color_ms, y=mag_ms,
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Approx. Main Sequence',
        hoverinfo='skip'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='BP - RP Color (mag)',
        yaxis_title='Absolute G Magnitude',
        yaxis=dict(autorange='reversed'),
        height=600,
        showlegend=True
    )
    
    return fig

def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">⭐ Astrophysical Oddball Finder</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Discover unusual stars in the Gaia DR3 dataset using machine learning**
    
    This dashboard allows you to explore stellar anomalies detected using multiple unsupervised ML methods:
    - **Isolation Forest**: Identifies outliers in high-dimensional feature space
    - **Autoencoder**: Detects objects with high reconstruction error
    - **DBSCAN**: Finds objects that don't belong to any cluster
    """)
    
    # Load data
    with st.spinner("Loading data..."):
        results, processed_data = load_data()
        anomaly_scores = load_anomaly_scores()
    
    if not results:
        st.error("""
        **No results found!** 
        
        Please run the complete pipeline first:
        1. `python src/data_acquisition.py` - Download Gaia data
        2. `python src/preprocessing.py` - Preprocess data
        3. `python src/models.py` - Train anomaly detection models
        
        Then restart this dashboard.
        """)
        return
    
    # Sidebar controls
    st.sidebar.header("🔧 Controls")
    
    # Method selection
    available_methods = list(results.keys())
    selected_method = st.sidebar.selectbox(
        "Select Anomaly Detection Method:",
        available_methods,
        index=0 if 'combined' in available_methods else 0
    )
    
    # Number of anomalies to show
    max_anomalies = len(results[selected_method]) if selected_method in results else 1000
    n_anomalies = st.sidebar.slider(
        "Number of Top Anomalies to Display:",
        min_value=10,
        max_value=min(1000, max_anomalies),
        value=min(100, max_anomalies),
        step=10
    )
    
    # Get selected results
    selected_results = results[selected_method].head(n_anomalies)
    
    # Main content area
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Sources Analyzed", 
            f"{len(processed_data):,}" if processed_data is not None else "N/A"
        )
    
    with col2:
        st.metric(
            "Anomalies Found", 
            f"{len(selected_results):,}"
        )
    
    with col3:
        st.metric(
            "Detection Method", 
            selected_method.replace('_', ' ').title()
        )
    
    with col4:
        avg_score = selected_results['anomaly_score'].mean()
        st.metric(
            "Avg Anomaly Score", 
            f"{avg_score:.4f}"
        )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🌟 HR Diagram", "📊 Anomaly List", "📈 Statistics", "🔍 Source Details"])
    
    with tab1:
        st.subheader("Interactive Hertzsprung-Russell Diagram")
        
        if processed_data is not None:
            # Source selection for highlighting
            source_to_highlight = st.selectbox(
                "Highlight specific source (optional):",
                options=[None] + selected_results['source_id'].head(20).tolist(),
                format_func=lambda x: "None" if x is None else f"Source {x}"
            )
            
            # Create HR diagram
            hr_fig = create_hr_diagram(
                processed_data, 
                selected_results, 
                source_to_highlight
            )
            st.plotly_chart(hr_fig, use_container_width=True)
            
            st.info("""
            **How to interpret this diagram:**
            - Normal stars follow the main sequence (red dashed line)
            - Anomalies (stars) are colored by their anomaly score
            - Click on points to see detailed information
            - Blue dots represent the general stellar population
            """)
        else:
            st.warning("HR Diagram requires processed data. Please run the data acquisition pipeline.")
    
    with tab2:
        st.subheader(f"Top {n_anomalies} Anomalies ({selected_method})")
        
        # Display options
        show_all_columns = st.checkbox("Show all columns", value=False)
        
        if show_all_columns:
            display_df = selected_results
        else:
            # Show key columns only
            key_columns = ['rank', 'source_id', 'ra', 'dec', 'anomaly_score']
            if 'v_tan_km_s' in selected_results.columns:
                key_columns.append('v_tan_km_s')
            if 'abs_g_mag' in selected_results.columns:
                key_columns.append('abs_g_mag')
            if 'bp_rp' in selected_results.columns:
                key_columns.append('bp_rp')
            
            available_columns = [col for col in key_columns if col in selected_results.columns]
            display_df = selected_results[available_columns]
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = selected_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Results as CSV",
            data=csv,
            file_name=f"anomalies_{selected_method}.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.subheader("Anomaly Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score distribution
            fig_hist = px.histogram(
                selected_results, 
                x='anomaly_score',
                nbins=30,
                title="Anomaly Score Distribution"
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Feature analysis if available
            if 'v_tan_km_s' in selected_results.columns:
                high_velocity = selected_results[selected_results['v_tan_km_s'] > 100]
                
                st.markdown("**Special Categories:**")
                st.write(f"🚀 High Velocity (>100 km/s): {len(high_velocity)}")
                
                if 'abs_g_mag' in selected_results.columns:
                    bright_objects = selected_results[selected_results['abs_g_mag'] < 0]
                    st.write(f"💫 Very Bright (Abs G < 0): {len(bright_objects)}")
                
                # Top velocity objects
                if len(high_velocity) > 0:
                    st.markdown("**Highest Velocity Objects:**")
                    top_velocity = high_velocity.nlargest(5, 'v_tan_km_s')
                    for _, row in top_velocity.iterrows():
                        st.write(f"Source {row['source_id']}: {row['v_tan_km_s']:.1f} km/s")
    
    with tab4:
        st.subheader("Individual Source Analysis")
        
        # Source selection
        selected_source_id = st.selectbox(
            "Select a source to analyze:",
            options=selected_results['source_id'].head(50).tolist()
        )
        
        if selected_source_id:
            source_data = selected_results[selected_results['source_id'] == selected_source_id].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Basic Information:**")
                st.write(f"**Source ID:** {source_data['source_id']}")
                st.write(f"**RA:** {source_data['ra']:.6f}°")
                st.write(f"**Dec:** {source_data['dec']:.6f}°")
                st.write(f"**Anomaly Rank:** {source_data['rank']}")
                st.write(f"**Anomaly Score:** {source_data['anomaly_score']:.6f}")
            
            with col2:
                st.markdown("**Physical Properties:**")
                if 'abs_g_mag' in source_data:
                    st.write(f"**Absolute G Mag:** {source_data['abs_g_mag']:.3f}")
                if 'bp_rp' in source_data:
                    st.write(f"**BP-RP Color:** {source_data['bp_rp']:.3f}")
                if 'v_tan_km_s' in source_data:
                    st.write(f"**Tangential Velocity:** {source_data['v_tan_km_s']:.1f} km/s")
                if 'distance_pc' in source_data:
                    st.write(f"**Distance:** {source_data['distance_pc']:.1f} pc")
            
            # SIMBAD link (if we had cross-matching)
            st.markdown(f"""
            **External Resources:**
            - [SIMBAD Query](http://simbad.u-strasbg.fr/simbad/sim-coo?Coord={source_data['ra']}+{source_data['dec']}&CooFrame=FK5&CooEpoch=2000&CooEqui=2000&CooDefinedFrames=none&Radius=10&Radius.unit=arcsec&submit=submit+query)
            - [Gaia Archive](https://gea.esac.esa.int/archive/)
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this project:** This dashboard showcases stellar anomalies detected in Gaia DR3 data using 
    unsupervised machine learning. The goal is to identify unusual stars that may represent rare 
    astrophysical phenomena or previously unknown stellar populations.
    
    **Methods used:** Isolation Forest, Autoencoder Neural Networks, DBSCAN Clustering
    """)

if __name__ == "__main__":
    main()
