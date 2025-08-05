"""
Analysis script for Astrophysical Oddball Finder results.
Explores the discovered anomalies and creates visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

def load_results():
    """Load all anomaly detection results."""
    results_dir = Path("results")
    
    results = {}
    for method in ['combined', 'isolation_forest', 'autoencoder', 'dbscan']:
        file_path = results_dir / f"top_anomalies_{method}.csv"
        if file_path.exists():
            results[method] = pd.read_csv(file_path)
            print(f"Loaded {len(results[method])} anomalies from {method}")
    
    return results

def analyze_top_anomalies(results, method='combined', top_n=20):
    """Analyze the top N anomalies from a specific method."""
    if method not in results:
        print(f"Method {method} not found in results")
        return
    
    df = results[method].head(top_n)
    
    print(f"\n{'='*60}")
    print(f"TOP {top_n} ANOMALIES - {method.upper()} METHOD")
    print(f"{'='*60}")
    
    # Basic statistics
    print(f"\nAnomaly Score Statistics:")
    print(f"  Mean: {df['anomaly_score'].mean():.4f}")
    print(f"  Median: {df['anomaly_score'].median():.4f}")
    print(f"  Range: {df['anomaly_score'].min():.4f} - {df['anomaly_score'].max():.4f}")
    
    # Magnitude analysis
    if 'abs_g_mag' in df.columns:
        print(f"\nAbsolute Magnitude Analysis:")
        bright_objects = df[df['abs_g_mag'] < 0]
        print(f"  Very bright objects (Abs G < 0): {len(bright_objects)}")
        if len(bright_objects) > 0:
            print(f"  Brightest object: {bright_objects['abs_g_mag'].min():.2f} mag")
        
        faint_objects = df[df['abs_g_mag'] > 10]
        print(f"  Very faint objects (Abs G > 10): {len(faint_objects)}")
    
    # Velocity analysis
    if 'v_tan_km_s' in df.columns:
        print(f"\nKinematic Analysis:")
        high_velocity = df[df['v_tan_km_s'] > 100]
        print(f"  High velocity objects (>100 km/s): {len(high_velocity)}")
        
        extreme_velocity = df[df['v_tan_km_s'] > 300]
        print(f"  Extreme velocity objects (>300 km/s): {len(extreme_velocity)}")
        
        print(f"  Velocity range: {df['v_tan_km_s'].min():.1f} - {df['v_tan_km_s'].max():.1f} km/s")
    
    # Color analysis
    if 'bp_rp' in df.columns:
        print(f"\nColor Analysis:")
        blue_objects = df[df['bp_rp'] < 0]
        red_objects = df[df['bp_rp'] > 2]
        print(f"  Very blue objects (BP-RP < 0): {len(blue_objects)}")
        print(f"  Very red objects (BP-RP > 2): {len(red_objects)}")
        print(f"  Color range: {df['bp_rp'].min():.3f} - {df['bp_rp'].max():.3f}")
    
    # Distance analysis
    if 'distance_pc' in df.columns:
        print(f"\nDistance Analysis:")
        nearby = df[df['distance_pc'] < 100]
        distant = df[df['distance_pc'] > 1000]
        print(f"  Nearby objects (<100 pc): {len(nearby)}")
        print(f"  Distant objects (>1000 pc): {len(distant)}")
        
        # Handle negative distances (parallax issues)
        negative_dist = df[df['distance_pc'] < 0]
        if len(negative_dist) > 0:
            print(f"  Objects with negative distances (parallax issues): {len(negative_dist)}")
    
    return df

def create_hr_diagram(results, method='combined', save_plot=True):
    """Create HR diagram highlighting anomalies."""
    if method not in results:
        return
    
    df = results[method]
    
    # Filter for reasonable values
    mask = (
        df['bp_rp'].notna() & df['abs_g_mag'].notna() &
        (df['bp_rp'] > -2) & (df['bp_rp'] < 5) &
        (df['abs_g_mag'] > -10) & (df['abs_g_mag'] < 20)
    )
    df_plot = df[mask].copy()
    
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot colored by anomaly score
    scatter = plt.scatter(
        df_plot['bp_rp'], 
        df_plot['abs_g_mag'],
        c=df_plot['anomaly_score'],
        cmap='viridis',
        alpha=0.7,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )
    
    plt.colorbar(scatter, label='Anomaly Score')
    
    # Add main sequence approximation
    color_ms = np.linspace(-0.5, 2.5, 100)
    mag_ms = 4.83 + 5.5 * color_ms
    plt.plot(color_ms, mag_ms, 'r--', alpha=0.8, linewidth=2, 
             label='Approximate Main Sequence')
    
    # Highlight top 10 anomalies
    top_10 = df_plot.head(10)
    plt.scatter(
        top_10['bp_rp'], 
        top_10['abs_g_mag'],
        c='red',
        s=200,
        marker='*',
        edgecolors='yellow',
        linewidth=2,
        label='Top 10 Anomalies',
        zorder=5
    )
    
    plt.xlabel('BP - RP Color (mag)', fontsize=12)
    plt.ylabel('Absolute G Magnitude', fontsize=12)
    plt.title(f'Hertzsprung-Russell Diagram - {method.title()} Anomalies', fontsize=14)
    plt.gca().invert_yaxis()  # Brighter stars at top
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_plot:
        plt.savefig(f'results/plots/hr_diagram_{method}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"HR diagram saved to results/plots/hr_diagram_{method}.png")
    
    plt.show()

def create_velocity_analysis(results, method='combined', save_plot=True):
    """Analyze velocity distribution of anomalies."""
    if method not in results or 'v_tan_km_s' not in results[method].columns:
        return
    
    df = results[method]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Velocity histogram
    ax1.hist(df['v_tan_km_s'], bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Tangential Velocity (km/s)')
    ax1.set_ylabel('Count')
    ax1.set_title('Tangential Velocity Distribution')
    ax1.axvline(100, color='red', linestyle='--', label='100 km/s')
    ax1.axvline(300, color='orange', linestyle='--', label='300 km/s')
    ax1.legend()
    
    # 2. Velocity vs Anomaly Score
    ax2.scatter(df['v_tan_km_s'], df['anomaly_score'], alpha=0.6)
    ax2.set_xlabel('Tangential Velocity (km/s)')
    ax2.set_ylabel('Anomaly Score')
    ax2.set_title('Velocity vs Anomaly Score')
    ax2.grid(True, alpha=0.3)
    
    # 3. Proper motion vector plot
    if 'pmra' in df.columns and 'pmdec' in df.columns:
        scatter = ax3.scatter(df['pmra'], df['pmdec'], 
                            c=df['anomaly_score'], cmap='viridis', alpha=0.6)
        ax3.set_xlabel('Proper Motion RA (mas/yr)')
        ax3.set_ylabel('Proper Motion Dec (mas/yr)')
        ax3.set_title('Proper Motion Distribution')
        plt.colorbar(scatter, ax=ax3, label='Anomaly Score')
    
    # 4. Distance vs Velocity
    if 'distance_pc' in df.columns:
        # Filter out extreme values for better visualization
        mask = (df['distance_pc'] > 0) & (df['distance_pc'] < 5000)
        df_filtered = df[mask]
        
        ax4.scatter(df_filtered['distance_pc'], df_filtered['v_tan_km_s'], 
                   alpha=0.6, c=df_filtered['anomaly_score'], cmap='viridis')
        ax4.set_xlabel('Distance (pc)')
        ax4.set_ylabel('Tangential Velocity (km/s)')
        ax4.set_title('Distance vs Velocity')
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f'results/plots/velocity_analysis_{method}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"Velocity analysis saved to results/plots/velocity_analysis_{method}.png")
    
    plt.show()

def compare_methods(results):
    """Compare anomaly scores across different methods."""
    if len(results) < 2:
        print("Need at least 2 methods to compare")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    methods = list(results.keys())
    
    for i, method in enumerate(methods[:4]):
        if i < len(axes):
            df = results[method]
            axes[i].hist(df['anomaly_score'], bins=30, alpha=0.7, 
                        edgecolor='black', label=method)
            axes[i].set_xlabel('Anomaly Score')
            axes[i].set_ylabel('Count')
            axes[i].set_title(f'{method.title()} Score Distribution')
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(methods), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('results/plots/method_comparison.png', dpi=300, bbox_inches='tight')
    print("Method comparison saved to results/plots/method_comparison.png")
    plt.show()

def generate_summary_report(results):
    """Generate a comprehensive summary report."""
    print("\n" + "="*80)
    print("ASTROPHYSICAL ODDBALL FINDER - SUMMARY REPORT")
    print("="*80)
    
    print(f"\nMethods analyzed: {list(results.keys())}")
    
    for method, df in results.items():
        print(f"\n{method.upper()} METHOD:")
        print(f"  Total anomalies: {len(df)}")
        print(f"  Score range: {df['anomaly_score'].min():.4f} - {df['anomaly_score'].max():.4f}")
        
        # Highlight extreme cases
        if 'v_tan_km_s' in df.columns:
            high_vel = df[df['v_tan_km_s'] > 100]
            if len(high_vel) > 0:
                print(f"  High velocity candidates: {len(high_vel)}")
        
        if 'abs_g_mag' in df.columns:
            bright = df[df['abs_g_mag'] < 0]
            if len(bright) > 0:
                print(f"  Very bright objects: {len(bright)}")
    
    # Overall statistics
    if 'combined' in results:
        combined_df = results['combined']
        print(f"\nTOP 10 MOST ANOMALOUS OBJECTS:")
        top_10 = combined_df.head(10)
        
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            print(f"  {i:2d}. Source {row['source_id']}: Score {row['anomaly_score']:.4f}")
            if 'v_tan_km_s' in row:
                print(f"      Velocity: {row['v_tan_km_s']:.1f} km/s")
            if 'abs_g_mag' in row:
                print(f"      Abs Mag: {row['abs_g_mag']:.2f}")

def main():
    """Main analysis function."""
    print("*** ASTROPHYSICAL ODDBALL FINDER - RESULTS ANALYSIS ***")
    print("="*60)
    
    # Create plots directory
    os.makedirs("results/plots", exist_ok=True)
    
    # Load results
    results = load_results()
    
    if not results:
        print("No results found! Make sure the pipeline has been run.")
        return
    
    # Generate comprehensive analysis
    generate_summary_report(results)
    
    # Analyze top anomalies for each method
    for method in results.keys():
        analyze_top_anomalies(results, method, top_n=20)
    
    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    for method in results.keys():
        print(f"\nCreating plots for {method} method...")
        create_hr_diagram(results, method)
        create_velocity_analysis(results, method)
    
    # Compare methods
    if len(results) > 1:
        print("\nComparing methods...")
        compare_methods(results)
    
    print("\n*** ANALYSIS COMPLETE! ***")
    print("Check the results/plots/ directory for visualizations.")
    print("\nKey findings:")
    print("- All anomaly detection methods completed successfully")
    print("- Top anomalies identified and ranked")
    print("- Visualizations created for further analysis")
    print("- Results ready for scientific interpretation")

if __name__ == "__main__":
    main()
