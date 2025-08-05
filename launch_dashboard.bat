@echo off
echo *** ASTROPHYSICAL ODDBALL FINDER - DASHBOARD LAUNCHER ***
echo ============================================================
echo.
echo Starting Streamlit dashboard...
echo Dashboard will open in your browser at: http://localhost:8501
echo Press Ctrl+C to stop the dashboard
echo.

cd dashboard
python -m streamlit run app.py --server.headless=true --server.port=8501 --browser.gatherUsageStats=false

pause
