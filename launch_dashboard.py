"""
Simple launcher script for the Astrophysical Oddball Finder dashboard.
This script ensures the dashboard launches correctly regardless of PATH issues.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit dashboard with proper error handling."""
    
    print("*** ASTROPHYSICAL ODDBALL FINDER - DASHBOARD LAUNCHER ***")
    print("="*60)
    
    # Check if we're in the right directory
    dashboard_dir = Path("dashboard")
    if not dashboard_dir.exists():
        print("[ERROR] Dashboard directory not found!")
        print("Make sure you're running this from the project root directory.")
        return False
    
    app_file = dashboard_dir / "app.py"
    if not app_file.exists():
        print("[ERROR] Dashboard app.py not found!")
        print(f"Expected location: {app_file}")
        return False
    
    # Check if results exist
    results_dir = Path("results")
    if not results_dir.exists():
        print("[WARNING] Results directory not found!")
        print("Run the main pipeline first: python main.py")
        print("Continuing anyway - dashboard will have limited functionality.")
    
    # Change to dashboard directory
    original_dir = os.getcwd()
    os.chdir(dashboard_dir)
    
    try:
        print("Starting Streamlit dashboard...")
        print("Dashboard will open in your browser at: http://localhost:8501")
        print("Press Ctrl+C to stop the dashboard")
        print("-" * 60)
        
        # Try different methods to launch streamlit
        methods = [
            [sys.executable, '-m', 'streamlit', 'run', 'app.py'],
            ['python', '-m', 'streamlit', 'run', 'app.py'],
            ['streamlit', 'run', 'app.py']
        ]
        
        for i, method in enumerate(methods, 1):
            try:
                print(f"Attempt {i}: {' '.join(method)}")
                subprocess.run(method, check=True)
                break  # If successful, break out of loop
                
            except FileNotFoundError as e:
                print(f"Method {i} failed: {e}")
                if i < len(methods):
                    print("Trying next method...")
                continue
                
            except subprocess.CalledProcessError as e:
                print(f"Method {i} failed with exit code {e.returncode}")
                if i < len(methods):
                    print("Trying next method...")
                continue
                
        else:
            # All methods failed
            print("\n[ERROR] All launch methods failed!")
            print("\nManual launch instructions:")
            print("1. Open a terminal/command prompt")
            print("2. Navigate to the dashboard directory:")
            print(f"   cd {Path.cwd()}")
            print("3. Run one of these commands:")
            print("   python -m streamlit run app.py")
            print("   OR")
            print("   streamlit run app.py")
            print("\nIf streamlit is not installed:")
            print("   pip install streamlit")
            return False
            
    except KeyboardInterrupt:
        print("\n[STOP] Dashboard stopped by user")
        return True
        
    finally:
        # Return to original directory
        os.chdir(original_dir)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
