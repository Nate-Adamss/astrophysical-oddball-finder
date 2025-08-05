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
import io
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import DBSCAN
try:
    from astroquery.simbad import Simbad
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    SIMBAD_AVAILABLE = True
except ImportError:
    SIMBAD_AVAILABLE = False
    st.warning("astroquery not available. SIMBAD cross-matching will be limited.")

# Add src to path
sys.path.append('../src')
from visualization import StellarVisualization

# Page configuration
st.set_page_config(
    page_title="Astrophysical Oddball Finder",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with Astronomy Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #0c1445 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #0c1445 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Animated Starfield Background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #eee, transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.8), transparent),
            radial-gradient(1px 1px at 90px 40px, #fff, transparent),
            radial-gradient(1px 1px at 130px 80px, rgba(255,255,255,0.6), transparent),
            radial-gradient(2px 2px at 160px 30px, #fff, transparent);
        background-repeat: repeat;
        background-size: 200px 100px;
        animation: sparkle 20s linear infinite;
        opacity: 0.4;
        z-index: -1;
        pointer-events: none;
    }
    
    @keyframes sparkle {
        from { transform: translateY(0px); }
        to { transform: translateY(-100px); }
    }
    
    /* Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #64b5f6, #42a5f5, #2196f3, #1976d2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 20px rgba(33, 150, 243, 0.3);
        animation: glow 3s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 10px rgba(33, 150, 243, 0.3)); }
        to { filter: drop-shadow(0 0 20px rgba(33, 150, 243, 0.6)); }
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: rgba(13, 17, 23, 0.95);
        border-right: 1px solid rgba(100, 181, 246, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Content Area */
    .main .block-container {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(100, 181, 246, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-top: 1rem;
        padding: 2rem;
    }
    
    /* Metric Cards */
    .metric-container {
        background: linear-gradient(135deg, rgba(100, 181, 246, 0.1), rgba(33, 150, 243, 0.05));
        border: 1px solid rgba(100, 181, 246, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(100, 181, 246, 0.3);
        border-color: rgba(100, 181, 246, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(13, 17, 23, 0.8);
        border-radius: 15px;
        padding: 0.5rem;
        border: 1px solid rgba(100, 181, 246, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #2196f3, #1976d2) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.4);
    }
    
    /* DataFrames */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(100, 181, 246, 0.2);
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #2196f3, #1976d2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(33, 150, 243, 0.5);
        background: linear-gradient(45deg, #1976d2, #1565c0);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(45deg, #4caf50, #388e3c);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.5);
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        background: rgba(100, 181, 246, 0.1);
        border: 1px solid rgba(100, 181, 246, 0.3);
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    
    /* Text Colors */
    .stMarkdown, .stText {
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Selectbox and Slider */
    .stSelectbox > div > div {
        background: rgba(13, 17, 23, 0.8);
        border: 1px solid rgba(100, 181, 246, 0.3);
        border-radius: 10px;
        color: white;
    }
    
    .stSlider {
        color: #2196f3;
    }
    
    /* Plotly Charts */
    .js-plotly-plot {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(13, 17, 23, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #2196f3, #1976d2);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #1976d2, #1565c0);
    }
    
    /* Header with cosmic effect */
    .cosmic-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, rgba(100, 181, 246, 0.1), rgba(33, 150, 243, 0.05));
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(100, 181, 246, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .cosmic-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(100, 181, 246, 0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .cosmic-title {
        position: relative;
        z-index: 1;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #64b5f6, #42a5f5, #2196f3, #1976d2, #9c27b0, #673ab7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .cosmic-subtitle {
        position: relative;
        z-index: 1;
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 300;
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

def create_simbad_url(ra, dec, radius=2):
    """Create SIMBAD coordinate search URL."""
    return f"https://simbad.u-strasbg.fr/simbad/sim-coo?Coord={ra}+{dec}&Radius={radius}&Radius.unit=arcsec"

def add_simbad_links(df):
    """Add SIMBAD links to dataframe."""
    if 'ra' in df.columns and 'dec' in df.columns:
        # Create clickable links using HTML
        df['View in SIMBAD'] = df.apply(
            lambda row: f'<a href="{create_simbad_url(row["ra"], row["dec"])}" target="_blank">🔗 SIMBAD</a>',
            axis=1
        )
    return df

def validate_gaia_columns(df):
    """Validate that uploaded dataframe contains required Gaia DR3 columns."""
    required_columns = ['source_id', 'ra', 'dec', 'parallax', 'pmra', 'pmdec', 'phot_g_mean_mag', 'bp_rp']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, missing_columns
    return True, []

def create_sample_data():
    """Create sample Gaia DR3 data for download."""
    sample_data = {
        'source_id': [1234567890123456789, 9876543210987654321, 5555555555555555555],
        'ra': [123.456789, 234.567890, 345.678901],
        'dec': [45.123456, -30.987654, 12.345678],
        'parallax': [10.5, 2.3, 0.8],
        'pmra': [15.2, -8.7, 3.1],
        'pmdec': [-12.8, 22.4, -5.9],
        'phot_g_mean_mag': [12.5, 15.8, 18.2],
        'bp_rp': [0.8, 1.2, 1.6]
    }
    return pd.DataFrame(sample_data)

def cross_match_simbad(df, max_sources=100):
    """Cross-match sources with SIMBAD database."""
    if not SIMBAD_AVAILABLE:
        df['simbad_type'] = 'N/A (astroquery not available)'
        return df
    
    # Limit to prevent timeout
    df_subset = df.head(max_sources).copy()
    df_subset['simbad_type'] = 'Unknown'
    
    # Configure SIMBAD query
    simbad = Simbad()
    simbad.add_votable_fields('otype')
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (idx, row) in enumerate(df_subset.iterrows()):
        try:
            # Update progress
            progress = (i + 1) / len(df_subset)
            progress_bar.progress(progress)
            status_text.text(f"Cross-matching with SIMBAD: {i+1}/{len(df_subset)} sources")
            
            # Query SIMBAD
            coord = SkyCoord(ra=row['ra']*u.degree, dec=row['dec']*u.degree, frame='icrs')
            result = simbad.query_region(coord, radius=2*u.arcsec)
            
            if result and len(result) > 0:
                # Get the closest match
                obj_type = result['OTYPE'][0] if 'OTYPE' in result.colnames else 'Unknown'
                df_subset.loc[idx, 'simbad_type'] = str(obj_type)
            else:
                df_subset.loc[idx, 'simbad_type'] = 'Not found'
                
            # Small delay to avoid overwhelming SIMBAD
            time.sleep(0.1)
            
        except Exception as e:
            df_subset.loc[idx, 'simbad_type'] = f'Error: {str(e)[:20]}...'
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    # For sources beyond max_sources, mark as not queried
    if len(df) > max_sources:
        remaining_df = df.iloc[max_sources:].copy()
        remaining_df['simbad_type'] = 'Not queried (limit reached)'
        df_subset = pd.concat([df_subset, remaining_df], ignore_index=True)
    
    return df_subset

def run_anomaly_detection_on_upload(df, threshold_percentile=99):
    """Run anomaly detection on uploaded data."""
    # Prepare features (similar to preprocessing.py)
    feature_columns = ['parallax', 'pmra', 'pmdec', 'phot_g_mean_mag', 'bp_rp']
    
    # Remove rows with NaN values in key columns
    df_clean = df.dropna(subset=feature_columns).copy()
    
    if len(df_clean) == 0:
        st.error("No valid data remaining after removing NaN values.")
        return df
    
    # Add derived features
    df_clean['abs_g_mag'] = df_clean['phot_g_mean_mag'] + 5 * np.log10(df_clean['parallax']/100)
    df_clean['v_tan_km_s'] = 4.74 * np.sqrt(df_clean['pmra']**2 + df_clean['pmdec']**2) / df_clean['parallax']
    
    # Select features for ML
    ml_features = ['parallax', 'pmra', 'pmdec', 'phot_g_mean_mag', 'bp_rp', 'abs_g_mag', 'v_tan_km_s']
    X = df_clean[ml_features].fillna(df_clean[ml_features].median())
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run Isolation Forest
    iso_forest = IsolationForest(contamination=(100-threshold_percentile)/100, random_state=42)
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    anomaly_scores = iso_forest.decision_function(X_scaled)
    
    # Add results to dataframe
    df_clean['anomaly_score'] = anomaly_scores
    df_clean['is_anomaly'] = anomaly_labels == -1
    
    # Rank by anomaly score
    df_clean = df_clean.sort_values('anomaly_score', ascending=True).reset_index(drop=True)
    df_clean['rank'] = range(1, len(df_clean) + 1)
    
    return df_clean

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
    
    # Enhanced Cosmic Header
    st.markdown("""
    <div class="cosmic-header">
        <h1 class="cosmic-title">🌌 Astrophysical Oddball Finder</h1>
        <p class="cosmic-subtitle">Discover unusual stellar objects in Gaia DR3 using machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced description with better styling
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(100, 181, 246, 0.05), rgba(33, 150, 243, 0.02));
        border: 1px solid rgba(100, 181, 246, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    ">
        <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.1rem; margin-bottom: 1rem; text-align: center;">
            🔬 <strong>Advanced ML Pipeline for Stellar Anomaly Detection</strong>
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🌲</div>
                <strong style="color: #64b5f6;">Isolation Forest</strong><br>
                <span style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;">High-dimensional outlier detection</span>
            </div>
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧠</div>
                <strong style="color: #42a5f5;">Autoencoder</strong><br>
                <span style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;">Reconstruction error analysis</span>
            </div>
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                <strong style="color: #2196f3;">DBSCAN</strong><br>
                <span style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;">Density-based clustering</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    # Enhanced Sidebar controls
    st.sidebar.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(100, 181, 246, 0.1), rgba(33, 150, 243, 0.05));
        border: 1px solid rgba(100, 181, 246, 0.3);
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
        text-align: center;
    ">
        <h2 style="color: #64b5f6; margin: 0; font-size: 1.5rem;">🎛️ Mission Control</h2>
        <p style="color: rgba(255, 255, 255, 0.7); margin: 0.5rem 0 0 0; font-size: 0.9rem;">Configure your anomaly search</p>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    # Enhanced Main content area with cosmic metrics
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(100, 181, 246, 0.05), rgba(33, 150, 243, 0.02));
        border: 1px solid rgba(100, 181, 246, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    ">
        <h3 style="color: #64b5f6; text-align: center; margin-bottom: 1.5rem;">📊 Mission Statistics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sources = len(processed_data) if processed_data is not None else 0
        st.markdown(f"""
        <div class="metric-container">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🌌</div>
                <div style="font-size: 2rem; font-weight: 700; color: #64b5f6; margin-bottom: 0.25rem;">{total_sources:,}</div>
                <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem;">Total Sources Analyzed</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎯</div>
                <div style="font-size: 2rem; font-weight: 700; color: #42a5f5; margin-bottom: 0.25rem;">{len(selected_results):,}</div>
                <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem;">Anomalies Displayed</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if selected_results is not None and len(selected_results) > 0:
            max_score = selected_results['anomaly_score'].max()
            score_display = f"{max_score:.3f}"
        else:
            score_display = "N/A"
        
        st.markdown(f"""
        <div class="metric-container">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚡</div>
                <div style="font-size: 2rem; font-weight: 700; color: #2196f3; margin-bottom: 0.25rem;">{score_display}</div>
                <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem;">Highest Anomaly Score</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        method_display = selected_method.replace('_', ' ').title()
        method_icons = {
            'Combined': '🎆',
            'Isolation Forest': '🌲',
            'Autoencoder': '🧠',
            'Dbscan': '🎯'
        }
        icon = method_icons.get(method_display, '🔬')
        
        st.markdown(f"""
        <div class="metric-container">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
                <div style="font-size: 1.2rem; font-weight: 600; color: #1976d2; margin-bottom: 0.25rem;">{method_display}</div>
                <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem;">Detection Method</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌟 HR Diagram", "📊 Anomaly List", "📈 Statistics", "🔍 Source Details", "📤 Upload Data"])
    
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
        
        # Create a copy of selected_results for display
        display_results = selected_results.copy()
        
        # Add SIMBAD links
        if 'ra' in display_results.columns and 'dec' in display_results.columns:
            display_results['View in SIMBAD'] = display_results.apply(
                lambda row: f"https://simbad.u-strasbg.fr/simbad/sim-coo?Coord={row['ra']}+{row['dec']}&Radius=2&Radius.unit=arcsec",
                axis=1
            )
        
        if show_all_columns:
            display_df = display_results
        else:
            # Show key columns only
            key_columns = ['rank', 'source_id', 'ra', 'dec', 'anomaly_score']
            if 'v_tan_km_s' in display_results.columns:
                key_columns.append('v_tan_km_s')
            if 'abs_g_mag' in display_results.columns:
                key_columns.append('abs_g_mag')
            if 'bp_rp' in display_results.columns:
                key_columns.append('bp_rp')
            
            # Always include SIMBAD column if available
            if 'View in SIMBAD' in display_results.columns:
                key_columns.append('View in SIMBAD')
            
            available_columns = [col for col in key_columns if col in display_results.columns]
            display_df = display_results[available_columns]
        
        # Display the dataframe with clickable links
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            column_config={
                "View in SIMBAD": st.column_config.LinkColumn(
                    "View in SIMBAD",
                    help="Click to open SIMBAD at this source's coordinates",
                    validate="^https://.*",
                    max_chars=100,
                    display_text="🔗 SIMBAD"
                )
            } if 'View in SIMBAD' in display_df.columns else None
        )
        
        # Information about SIMBAD integration
        if 'View in SIMBAD' in display_df.columns:
            st.info("""
            **🔗 SIMBAD Integration:** Click the SIMBAD links to instantly validate anomalies!
            Each link opens SIMBAD's coordinate search at the source's RA/Dec with a 2 arcsecond radius.
            This helps you quickly identify if the anomaly corresponds to a known astronomical object.
            """)
        
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
    
    with tab5:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(100, 181, 246, 0.05), rgba(33, 150, 243, 0.02));
            border: 1px solid rgba(100, 181, 246, 0.2);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        ">
            <h3 style="color: #64b5f6; text-align: center; margin-bottom: 1rem;">🚀 Upload Your Gaia DR3 Dataset</h3>
            <p style="color: rgba(255, 255, 255, 0.9); text-align: center; margin-bottom: 1rem;">
                Analyze your own stellar data with our advanced anomaly detection pipeline
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div style="
                background: rgba(100, 181, 246, 0.05);
                border: 1px solid rgba(100, 181, 246, 0.2);
                border-radius: 10px;
                padding: 1rem;
                margin-bottom: 1rem;
            ">
                <h4 style="color: #42a5f5; margin-bottom: 0.5rem;">📁 Required Columns</h4>
                <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; margin-bottom: 0.5rem;">
                    Your CSV file must contain these Gaia DR3 columns:
                </p>
                <ul style="color: rgba(255, 255, 255, 0.7); font-size: 0.85rem; margin: 0;">
                    <li><code>source_id</code> - Gaia source identifier</li>
                    <li><code>ra, dec</code> - Right ascension and declination (degrees)</li>
                    <li><code>parallax</code> - Parallax (mas)</li>
                    <li><code>pmra, pmdec</code> - Proper motion (mas/yr)</li>
                    <li><code>phot_g_mean_mag</code> - G-band magnitude</li>
                    <li><code>bp_rp</code> - BP-RP color index</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # File upload
            uploaded_file = st.file_uploader(
                "Choose your Gaia DR3 CSV file",
                type=['csv'],
                help="Upload a CSV file containing Gaia DR3 stellar data"
            )
        
        with col2:
            # Sample data download
            sample_df = create_sample_data()
            csv_sample = sample_df.to_csv(index=False)
            
            st.markdown("""
            <div style="
                background: rgba(76, 175, 80, 0.05);
                border: 1px solid rgba(76, 175, 80, 0.2);
                border-radius: 10px;
                padding: 1rem;
                text-align: center;
            ">
                <h4 style="color: #4caf50; margin-bottom: 0.5rem;">📎 Sample Data</h4>
                <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; margin-bottom: 1rem;">
                    Download a sample file to see the expected format
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.download_button(
                label="📥 Download Sample CSV",
                data=csv_sample,
                file_name="gaia_dr3_sample.csv",
                mime="text/csv",
                help="Download this sample to see the required format"
            )
        
        # Process uploaded file
        if uploaded_file is not None:
            try:
                # Read uploaded file
                user_df = pd.read_csv(uploaded_file)
                
                # Validate columns
                is_valid, missing_cols = validate_gaia_columns(user_df)
                
                if not is_valid:
                    st.error(f"""
                    **Missing required columns:** {', '.join(missing_cols)}
                    
                    Please ensure your CSV file contains all required Gaia DR3 columns.
                    Download the sample file above to see the expected format.
                    """)
                else:
                    st.success(f"✅ File validated successfully! Found {len(user_df):,} sources.")
                    
                    # Advanced controls
                    st.markdown("""
                    <div style="
                        background: rgba(156, 39, 176, 0.05);
                        border: 1px solid rgba(156, 39, 176, 0.2);
                        border-radius: 10px;
                        padding: 1rem;
                        margin: 1rem 0;
                    ">
                        <h4 style="color: #9c27b0; margin-bottom: 1rem;">⚙️ Analysis Settings</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        threshold_percentile = st.slider(
                            "Anomaly Threshold (Top %)",
                            min_value=90.0,
                            max_value=99.9,
                            value=99.0,
                            step=0.1,
                            help="Percentage of sources to consider as normal (higher = fewer anomalies)"
                        )
                    
                    with col2:
                        max_simbad_sources = st.number_input(
                            "Max SIMBAD Cross-matches",
                            min_value=10,
                            max_value=500,
                            value=100,
                            step=10,
                            help="Limit SIMBAD queries to prevent timeout"
                        )
                    
                    with col3:
                        expected_anomalies = int(len(user_df) * (100 - threshold_percentile) / 100)
                        st.metric(
                            "Expected Anomalies",
                            f"{expected_anomalies:,}",
                            help=f"Approximately {expected_anomalies} sources will be flagged as anomalies"
                        )
                    
                    # Process button
                    if st.button("🚀 Analyze Dataset", type="primary"):
                        with st.spinner("Running anomaly detection..."):
                            # Run anomaly detection
                            processed_df = run_anomaly_detection_on_upload(user_df, threshold_percentile)
                            
                            if len(processed_df) > 0:
                                # Get anomalies only
                                anomalies_df = processed_df[processed_df['is_anomaly']].copy()
                                
                                st.success(f"✅ Analysis complete! Found {len(anomalies_df):,} anomalies.")
                                
                                # Cross-match with SIMBAD
                                if len(anomalies_df) > 0 and SIMBAD_AVAILABLE:
                                    st.info("🔍 Starting SIMBAD cross-match...")
                                    anomalies_df = cross_match_simbad(anomalies_df, max_simbad_sources)
                                
                                # Add SIMBAD links
                                anomalies_df['View in SIMBAD'] = anomalies_df.apply(
                                    lambda row: f"https://simbad.u-strasbg.fr/simbad/sim-coo?Coord={row['ra']}+{row['dec']}&Radius=2&Radius.unit=arcsec",
                                    axis=1
                                )
                                
                                # Store in session state for persistence
                                st.session_state['user_anomalies'] = anomalies_df
                                st.session_state['user_full_data'] = processed_df
                    
                    # Display results if available
                    if 'user_anomalies' in st.session_state:
                        anomalies_df = st.session_state['user_anomalies']
                        processed_df = st.session_state['user_full_data']
                        
                        st.markdown("""
                        <div style="
                            background: linear-gradient(135deg, rgba(255, 193, 7, 0.1), rgba(255, 152, 0, 0.05));
                            border: 1px solid rgba(255, 193, 7, 0.3);
                            border-radius: 15px;
                            padding: 1.5rem;
                            margin: 2rem 0;
                        ">
                            <h3 style="color: #ffc107; text-align: center; margin-bottom: 1rem;">🎆 User Dataset Results</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Results tabs
                        result_tab1, result_tab2, result_tab3 = st.tabs(["📊 Anomaly Table", "🌟 HR Diagram", "📥 Export"])
                        
                        with result_tab1:
                            st.subheader(f"Top {len(anomalies_df)} Anomalies in Your Dataset")
                            
                            # Display columns selection
                            show_all_cols = st.checkbox("Show all columns", value=False, key="user_show_all")
                            
                            if show_all_cols:
                                display_cols = anomalies_df.columns.tolist()
                            else:
                                display_cols = ['rank', 'source_id', 'ra', 'dec', 'anomaly_score']
                                if 'simbad_type' in anomalies_df.columns:
                                    display_cols.append('simbad_type')
                                if 'View in SIMBAD' in anomalies_df.columns:
                                    display_cols.append('View in SIMBAD')
                                
                                # Add other key columns if available
                                for col in ['v_tan_km_s', 'abs_g_mag', 'bp_rp']:
                                    if col in anomalies_df.columns:
                                        display_cols.append(col)
                            
                            available_cols = [col for col in display_cols if col in anomalies_df.columns]
                            display_df = anomalies_df[available_cols]
                            
                            # Display with enhanced styling
                            st.dataframe(
                                display_df,
                                use_container_width=True,
                                height=400,
                                column_config={
                                    "View in SIMBAD": st.column_config.LinkColumn(
                                        "View in SIMBAD",
                                        help="Click to open SIMBAD at this source's coordinates",
                                        validate="^https://.*",
                                        max_chars=100,
                                        display_text="🔗 SIMBAD"
                                    )
                                } if 'View in SIMBAD' in display_df.columns else None
                            )
                            
                            # SIMBAD integration info
                            if 'simbad_type' in anomalies_df.columns:
                                st.info("""
                                🔗 **SIMBAD Cross-match Results:** Object types have been automatically retrieved from SIMBAD.
                                Click the SIMBAD links to explore each source in detail and validate the anomaly detection.
                                """)
                        
                        with result_tab2:
                            st.subheader("HR Diagram - Your Dataset")
                            
                            if len(processed_df) > 0:
                                # Create HR diagram for user data
                                hr_fig = create_hr_diagram(processed_df, anomalies_df)
                                st.plotly_chart(hr_fig, use_container_width=True)
                                
                                st.info("""
                                **Your Data HR Diagram:** Normal sources are shown in blue, anomalies are highlighted as stars.
                                The color intensity represents the anomaly score - darker colors indicate more unusual objects.
                                """)
                            else:
                                st.warning("No data available for HR diagram.")
                        
                        with result_tab3:
                            st.subheader("Export Your Results")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Export anomalies only
                                anomalies_csv = anomalies_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Anomalies CSV",
                                    data=anomalies_csv,
                                    file_name="user_dataset_anomalies.csv",
                                    mime="text/csv",
                                    help="Download only the detected anomalies with SIMBAD classifications"
                                )
                            
                            with col2:
                                # Export full processed dataset
                                full_csv = processed_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Full Processed Dataset",
                                    data=full_csv,
                                    file_name="user_dataset_processed.csv",
                                    mime="text/csv",
                                    help="Download the complete dataset with anomaly scores and derived features"
                                )
                            
                            # Summary statistics
                            st.markdown("""
                            <div style="
                                background: rgba(33, 150, 243, 0.05);
                                border: 1px solid rgba(33, 150, 243, 0.2);
                                border-radius: 10px;
                                padding: 1rem;
                                margin-top: 1rem;
                            ">
                                <h4 style="color: #2196f3; margin-bottom: 0.5rem;">📈 Analysis Summary</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            summary_col1, summary_col2, summary_col3 = st.columns(3)
                            
                            with summary_col1:
                                st.metric("Total Sources", f"{len(processed_df):,}")
                            
                            with summary_col2:
                                st.metric("Anomalies Found", f"{len(anomalies_df):,}")
                            
                            with summary_col3:
                                anomaly_rate = (len(anomalies_df) / len(processed_df)) * 100 if len(processed_df) > 0 else 0
                                st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
                            
                            if 'simbad_type' in anomalies_df.columns:
                                st.markdown("**SIMBAD Classifications:**")
                                simbad_counts = anomalies_df['simbad_type'].value_counts()
                                for obj_type, count in simbad_counts.head(10).items():
                                    st.text(f"• {obj_type}: {count} sources")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure your file is a valid CSV with the required Gaia DR3 columns.")
    
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
