# Astrophysical Oddball Finder

## Overview
This project aims to identify unusual stars or sources in the Gaia DR3 dataset using unsupervised machine learning techniques. The goal is to create a ranked list of anomalous objects and provide an interactive visualization dashboard for exploring these "oddball" stars.

## Problem Statement
Using Gaia DR3 astrometric and photometric data, we will:
- Apply unsupervised ML methods to detect stellar outliers
- Cross-validate findings with known catalogs (SIMBAD)
- Identify both known unusual stars and potential new candidates
- Create an interactive tool for exploring anomalies

## Success Metrics
- Rediscover a significant percentage of known unusual stars (hypervelocity stars, white dwarfs in unusual locations, etc.)
- Generate a ranked list of new anomaly candidates for follow-up
- Build a functional interactive dashboard for anomaly exploration

## Dataset
**Primary**: Gaia DR3 (European Space Agency)
- High-quality sources with filters:
  - `parallax_over_error > 5` (reliable distance measurements)
  - `ruwe < 1.4` (good astrometric fits)

**Secondary**: SIMBAD (cross-matching for validation)

## Key Features
- **Astrometric**: Parallax (distance), proper motion (μ_RA, μ_DEC)
- **Photometric**: G magnitude, BP-RP color
- **Derived**: Tangential velocity, color-magnitude position

## Machine Learning Approach
1. **Isolation Forest**: High-dimensional anomaly detection
2. **Autoencoder**: Neural network reconstruction error analysis
3. **DBSCAN/HDBSCAN**: Clustering-based outlier detection

## Tech Stack
- **Languages**: Python
- **Core Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn, plotly
- **Astronomy**: astroquery, astropy
- **Dashboard**: Streamlit or Plotly Dash
- **Version Control**: Git + GitHub

## Project Structure
```
astrophysical-oddball-finder/
├── data/                   # Raw and processed datasets
├── src/                    # Source code modules
│   ├── data_acquisition.py
│   ├── preprocessing.py
│   ├── models.py
│   └── visualization.py
├── notebooks/              # Jupyter notebooks for EDA
├── dashboard/              # Interactive dashboard code
├── results/                # Model outputs and findings
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Getting Started
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run data acquisition: `python src/data_acquisition.py`
4. Follow the notebooks for exploratory analysis
5. Train models and generate anomaly rankings
6. Launch the interactive dashboard

## Timeline
- **Weeks 1-3**: Data acquisition and preprocessing
- **Week 4**: Exploratory data analysis
- **Weeks 5-7**: Model development and training
- **Weeks 8-9**: Validation and cross-matching
- **Weeks 10-11**: Dashboard development
- **Weeks 12-13**: Documentation and publication

## Expected Outcomes
- Ranked list of stellar anomalies with confidence scores
- Interactive HR diagram highlighting unusual objects
- Validation results comparing with known catalogs
- Documentation of methodology and findings

## Contributing
This is a research project. Feel free to suggest improvements or report issues.

## License
MIT License - See LICENSE file for details
