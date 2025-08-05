"""
Visualization module for stellar anomaly detection results.
Creates plots for EDA and anomaly analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class StellarVisualization:
    """Class for creating stellar data visualizations."""
    
    def __init__(self, output_dir="../results/plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_hr_diagram(self, df, color_col='bp_rp', mag_col='abs_g_mag', 
                       anomaly_scores=None, title="Hertzsprung-Russell Diagram"):
        """
        Create HR diagram (Color-Magnitude diagram).
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Stellar data
        color_col : str
            Column for color (x-axis)
        mag_col : str
            Column for magnitude (y-axis)
        anomaly_scores : array-like
            Anomaly scores for coloring points
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Filter out extreme values for better visualization
        valid_mask = (
            np.isfinite(df[color_col]) & 
            np.isfinite(df[mag_col]) &
            (df[color_col] > -1) & (df[color_col] < 4) &
            (df[mag_col] > -5) & (df[mag_col] < 20)
        )
        
        df_plot = df[valid_mask].copy()
        
        if anomaly_scores is not None:
            scores_plot = anomaly_scores[valid_mask]
            scatter = ax.scatter(
                df_plot[color_col], df_plot[mag_col],
                c=scores_plot, cmap='viridis', alpha=0.6, s=1
            )
            plt.colorbar(scatter, label='Anomaly Score')
        else:
            ax.scatter(
                df_plot[color_col], df_plot[mag_col],
                alpha=0.6, s=1, color='blue'
            )
        
        ax.set_xlabel('BP - RP Color (mag)')
        ax.set_ylabel('Absolute G Magnitude')
        ax.set_title(title)
        ax.invert_yaxis()  # Brighter stars at top
        ax.grid(True, alpha=0.3)
        
        # Add main sequence guide line (approximate)
        color_ms = np.linspace(-0.5, 2.0, 100)
        mag_ms = 4.83 + 5.5 * color_ms  # Rough main sequence relation
        ax.plot(color_ms, mag_ms, 'r--', alpha=0.7, label='Approx. Main Sequence')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'hr_diagram.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig, ax
    
    def plot_proper_motion_distribution(self, df, anomaly_scores=None):
        """Plot proper motion distribution and vector field."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Proper motion magnitude distribution
        pm_total = np.sqrt(df['pmra']**2 + df['pmdec']**2)
        ax1.hist(pm_total, bins=50, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Total Proper Motion (mas/yr)')
        ax1.set_ylabel('Count')
        ax1.set_title('Proper Motion Magnitude Distribution')
        ax1.set_yscale('log')
        
        # 2. Proper motion vector plot
        if anomaly_scores is not None:
            scatter = ax2.scatter(df['pmra'], df['pmdec'], c=anomaly_scores, 
                                cmap='viridis', alpha=0.6, s=1)
            plt.colorbar(scatter, ax=ax2, label='Anomaly Score')
        else:
            ax2.scatter(df['pmra'], df['pmdec'], alpha=0.6, s=1)
        
        ax2.set_xlabel('Proper Motion RA (mas/yr)')
        ax2.set_ylabel('Proper Motion Dec (mas/yr)')
        ax2.set_title('Proper Motion Vector Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Tangential velocity distribution
        if 'v_tan_km_s' in df.columns:
            ax3.hist(df['v_tan_km_s'], bins=50, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Tangential Velocity (km/s)')
            ax3.set_ylabel('Count')
            ax3.set_title('Tangential Velocity Distribution')
            ax3.set_yscale('log')
        
        # 4. Distance vs proper motion
        if 'distance_pc' in df.columns:
            ax4.scatter(df['distance_pc'], pm_total, alpha=0.6, s=1)
            ax4.set_xlabel('Distance (pc)')
            ax4.set_ylabel('Total Proper Motion (mas/yr)')
            ax4.set_title('Distance vs Proper Motion')
            ax4.set_xscale('log')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'proper_motion_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_anomaly_score_distributions(self, anomaly_scores_dict):
        """Plot distributions of anomaly scores from different methods."""
        n_methods = len(anomaly_scores_dict)
        fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(15, 10))
        axes = axes.flatten() if n_methods > 1 else [axes]
        
        for i, (method, data) in enumerate(anomaly_scores_dict.items()):
            scores = data['scores']
            threshold = data.get('threshold', None)
            
            axes[i].hist(scores, bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel('Anomaly Score')
            axes[i].set_ylabel('Count')
            axes[i].set_title(f'{method.replace("_", " ").title()} Scores')
            axes[i].set_yscale('log')
            
            if threshold is not None:
                axes[i].axvline(threshold, color='red', linestyle='--', 
                               label=f'Threshold: {threshold:.3f}')
                axes[i].legend()
        
        # Hide unused subplots
        for i in range(len(anomaly_scores_dict), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'anomaly_score_distributions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_interactive_hr_diagram(self, df, anomaly_scores=None, 
                                    top_anomalies=None, save_html=True):
        """Create interactive HR diagram using Plotly."""
        
        # Prepare data
        valid_mask = (
            np.isfinite(df['bp_rp']) & 
            np.isfinite(df['abs_g_mag']) &
            (df['bp_rp'] > -1) & (df['bp_rp'] < 4) &
            (df['abs_g_mag'] > -5) & (df['abs_g_mag'] < 20)
        )
        
        df_plot = df[valid_mask].copy()
        
        # Create base scatter plot
        if anomaly_scores is not None:
            scores_plot = anomaly_scores[valid_mask]
            
            fig = px.scatter(
                x=df_plot['bp_rp'], 
                y=df_plot['abs_g_mag'],
                color=scores_plot,
                color_continuous_scale='viridis',
                opacity=0.6,
                hover_data=['source_id'] if 'source_id' in df_plot.columns else None,
                labels={
                    'x': 'BP - RP Color (mag)',
                    'y': 'Absolute G Magnitude',
                    'color': 'Anomaly Score'
                },
                title='Interactive Hertzsprung-Russell Diagram'
            )
        else:
            fig = px.scatter(
                x=df_plot['bp_rp'], 
                y=df_plot['abs_g_mag'],
                opacity=0.6,
                hover_data=['source_id'] if 'source_id' in df_plot.columns else None,
                labels={
                    'x': 'BP - RP Color (mag)',
                    'y': 'Absolute G Magnitude'
                },
                title='Interactive Hertzsprung-Russell Diagram'
            )
        
        # Highlight top anomalies if provided
        if top_anomalies is not None:
            fig.add_scatter(
                x=top_anomalies['bp_rp'],
                y=top_anomalies['abs_g_mag'],
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='star',
                    line=dict(width=1, color='black')
                ),
                name='Top Anomalies',
                hovertemplate='<b>Anomaly</b><br>' +
                             'Source ID: %{customdata[0]}<br>' +
                             'BP-RP: %{x:.3f}<br>' +
                             'Abs G Mag: %{y:.3f}<br>' +
                             'Anomaly Score: %{customdata[1]:.3f}',
                customdata=np.column_stack([
                    top_anomalies['source_id'].values,
                    top_anomalies['anomaly_score'].values
                ])
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
            yaxis=dict(autorange='reversed'),  # Brighter stars at top
            width=800,
            height=600,
            showlegend=True
        )
        
        if save_html:
            html_file = os.path.join(self.output_dir, 'interactive_hr_diagram.html')
            fig.write_html(html_file)
            print(f"Interactive HR diagram saved to: {html_file}")
        
        fig.show()
        return fig
    
    def plot_feature_correlations(self, df, anomaly_scores=None):
        """Plot correlation matrix of features."""
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_correlations.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig, corr_matrix
    
    def create_anomaly_summary_report(self, top_anomalies, method='combined'):
        """Create a summary report of top anomalies."""
        print(f"\n=== Top Anomalies Summary ({method}) ===")
        print(f"Total anomalies analyzed: {len(top_anomalies)}")
        
        # Basic statistics
        print(f"\nAnomaly Score Statistics:")
        print(f"  Mean: {top_anomalies['anomaly_score'].mean():.4f}")
        print(f"  Median: {top_anomalies['anomaly_score'].median():.4f}")
        print(f"  Max: {top_anomalies['anomaly_score'].max():.4f}")
        print(f"  Min: {top_anomalies['anomaly_score'].min():.4f}")
        
        # Top 10 anomalies
        print(f"\nTop 10 Most Anomalous Sources:")
        print(top_anomalies.head(10)[['rank', 'source_id', 'ra', 'dec', 'anomaly_score']].to_string(index=False))
        
        # Feature distributions for top anomalies
        if 'v_tan_km_s' in top_anomalies.columns:
            high_velocity = top_anomalies[top_anomalies['v_tan_km_s'] > 100]
            print(f"\nHigh Velocity Candidates (>100 km/s): {len(high_velocity)}")
        
        if 'abs_g_mag' in top_anomalies.columns:
            bright_objects = top_anomalies[top_anomalies['abs_g_mag'] < 0]
            print(f"Very Bright Objects (Abs G < 0): {len(bright_objects)}")
        
        return top_anomalies.head(10)

def main():
    """Main function for creating visualizations."""
    print("=== Creating Stellar Visualizations ===")
    
    # This would typically be called after model training
    # For now, just set up the visualization class
    viz = StellarVisualization()
    print(f"Visualization output directory: {viz.output_dir}")
    
    print("Visualization module ready!")
    print("Use this module after running the full pipeline to create plots.")

if __name__ == "__main__":
    main()
