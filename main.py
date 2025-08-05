"""
Main execution script for the Astrophysical Oddball Finder project.
Orchestrates the complete pipeline from data acquisition to dashboard.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"{'='*50}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("[OK] SUCCESS")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("[ERROR] FAILED")
        print("Error:", e.stderr)
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'seaborn', 'plotly', 'astroquery', 'streamlit'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"[ERROR] Missing packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("[OK] All dependencies satisfied")
        return True

def run_data_acquisition(max_sources=100000):
    """Run data acquisition step."""
    os.chdir('src')
    success = run_command(
        f'python data_acquisition.py',
        f"Data Acquisition (downloading {max_sources:,} Gaia DR3 sources)"
    )
    os.chdir('..')
    return success

def run_preprocessing():
    """Run data preprocessing step."""
    os.chdir('src')
    success = run_command(
        'python preprocessing.py',
        "Data Preprocessing (cleaning and feature engineering)"
    )
    os.chdir('..')
    return success

def run_model_training():
    """Run model training step."""
    os.chdir('src')
    success = run_command(
        'python models.py',
        "Model Training (Isolation Forest, Autoencoder, DBSCAN)"
    )
    os.chdir('..')
    return success

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    os.chdir('dashboard')
    print(f"\n{'='*50}")
    print("LAUNCHING DASHBOARD")
    print(f"{'='*50}")
    print("Starting Streamlit dashboard...")
    print("Dashboard will open in your browser at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    
    try:
        # Use python -m streamlit instead of direct streamlit command
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n[STOP] Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to launch dashboard: {e}")
        print("\nAlternative: Try running manually:")
        print("  cd dashboard")
        print("  python -m streamlit run app.py")
    except FileNotFoundError as e:
        print(f"[ERROR] Streamlit not found: {e}")
        print("\nTry installing streamlit:")
        print("  pip install streamlit")
        print("\nThen run manually:")
        print("  cd dashboard")
        print("  python -m streamlit run app.py")
    finally:
        os.chdir('..')

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Astrophysical Oddball Finder - Complete Pipeline"
    )
    parser.add_argument(
        '--step', 
        choices=['all', 'data', 'preprocess', 'train', 'dashboard'],
        default='all',
        help='Which step to run (default: all)'
    )
    parser.add_argument(
        '--max-sources', 
        type=int, 
        default=100000,
        help='Maximum number of Gaia sources to download (default: 100000)'
    )
    parser.add_argument(
        '--skip-deps', 
        action='store_true',
        help='Skip dependency check'
    )
    
    args = parser.parse_args()
    
    print("*** ASTROPHYSICAL ODDBALL FINDER ***")
    print("Discovering unusual stars with machine learning")
    print(f"Project directory: {os.getcwd()}")
    
    # Check dependencies
    if not args.skip_deps:
        if not check_dependencies():
            print("\n[ERROR] Dependency check failed. Install requirements first:")
            print("pip install -r requirements.txt")
            return False
    
    # Create directories if they don't exist
    for directory in ['data', 'results', 'results/plots']:
        os.makedirs(directory, exist_ok=True)
    
    success = True
    
    # Run selected steps
    if args.step in ['all', 'data']:
        success &= run_data_acquisition(args.max_sources)
        if not success:
            print("[ERROR] Data acquisition failed. Stopping pipeline.")
            return False
    
    if args.step in ['all', 'preprocess']:
        if not os.path.exists('data/gaia_dr3_processed.csv'):
            print("[ERROR] Processed data not found. Run data acquisition first.")
            return False
        success &= run_preprocessing()
        if not success:
            print("[ERROR] Preprocessing failed. Stopping pipeline.")
            return False
    
    if args.step in ['all', 'train']:
        if not os.path.exists('data/training_features.csv'):
            print("[ERROR] Training data not found. Run preprocessing first.")
            return False
        success &= run_model_training()
        if not success:
            print("[ERROR] Model training failed. Stopping pipeline.")
            return False
    
    if args.step in ['all', 'dashboard']:
        if not os.path.exists('results/top_anomalies_combined.csv'):
            print("[WARNING] No anomaly results found. Dashboard will have limited functionality.")
        launch_dashboard()
    
    if success and args.step == 'all':
        print(f"\n{'='*50}")
        print("*** PIPELINE COMPLETE! ***")
        print(f"{'='*50}")
        print("[OK] Data acquisition completed")
        print("[OK] Preprocessing completed") 
        print("[OK] Model training completed")
        print("[OK] Dashboard launched")
        print("\nNext steps:")
        print("1. Explore anomalies in the dashboard")
        print("2. Cross-match with SIMBAD for validation")
        print("3. Analyze interesting candidates")
        print("4. Document findings in notebooks/")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
