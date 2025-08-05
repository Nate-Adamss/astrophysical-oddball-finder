#!/usr/bin/env python3
"""
Simple dashboard runner for Astrophysical Oddball Finder.
This bypasses PATH issues by running streamlit directly through Python.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the dashboard with proper setup."""
    print("*** ASTROPHYSICAL ODDBALL FINDER - DASHBOARD ***")
    print("=" * 55)
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    dashboard_dir = project_root / "dashboard"
    
    if not dashboard_dir.exists():
        print("[ERROR] Dashboard directory not found!")
        return False
    
    # Change to dashboard directory
    os.chdir(dashboard_dir)
    
    print("Starting interactive dashboard...")
    print("Opening browser at: http://localhost:8501")
    print("Press Ctrl+C to stop")
    print("-" * 55)
    
    try:
        # Run streamlit with explicit configuration
        cmd = [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--server.enableCORS=false"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n[STOP] Dashboard stopped successfully!")
        return True
    except Exception as e:
        print(f"\n[ERROR] Error running dashboard: {e}")
        print("\nManual instructions:")
        print("1. Open terminal/command prompt")
        print(f"2. cd {dashboard_dir}")
        print("3. python -m streamlit run app.py")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
