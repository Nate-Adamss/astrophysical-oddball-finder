"""
Data acquisition module for Gaia DR3 data.
Handles downloading and initial filtering of stellar data.
"""

import pandas as pd
import numpy as np
from astroquery.gaia import Gaia
from astropy.table import Table
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class GaiaDataAcquisition:
    """Class for acquiring Gaia DR3 data with quality filters."""
    
    def __init__(self, output_dir="../data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up Gaia query service
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        Gaia.ROW_LIMIT = -1  # Remove row limit for large queries
    
    def build_quality_query(self, max_sources=1000000):
        """
        Build ADQL query for high-quality Gaia sources.
        
        Parameters:
        -----------
        max_sources : int
            Maximum number of sources to retrieve
            
        Returns:
        --------
        str : ADQL query string
        """
        query = f"""
        SELECT TOP {max_sources}
            source_id,
            ra, dec,
            parallax, parallax_error,
            pmra, pmra_error,
            pmdec, pmdec_error,
            phot_g_mean_mag,
            phot_bp_mean_mag,
            phot_rp_mean_mag,
            bp_rp,
            ruwe,
            astrometric_excess_noise,
            visibility_periods_used
        FROM gaiadr3.gaia_source
        WHERE 
            parallax_over_error > 5.0
            AND ruwe < 1.4
            AND phot_g_mean_mag IS NOT NULL
            AND phot_bp_mean_mag IS NOT NULL
            AND phot_rp_mean_mag IS NOT NULL
            AND pmra IS NOT NULL
            AND pmdec IS NOT NULL
            AND parallax > 0
            AND astrometric_excess_noise < 1.0
            AND visibility_periods_used > 8
        ORDER BY phot_g_mean_mag
        """
        return query
    
    def download_gaia_data(self, max_sources=1000000, save_raw=True):
        """
        Download Gaia DR3 data with quality filters.
        
        Parameters:
        -----------
        max_sources : int
            Maximum number of sources to retrieve
        save_raw : bool
            Whether to save raw downloaded data
            
        Returns:
        --------
        pandas.DataFrame : Downloaded Gaia data
        """
        print(f"Downloading up to {max_sources:,} high-quality Gaia DR3 sources...")
        print("Quality filters applied:")
        print("  - parallax_over_error > 5.0 (reliable distances)")
        print("  - ruwe < 1.4 (good astrometric fits)")
        print("  - Complete photometry (G, BP, RP)")
        print("  - Complete proper motions")
        print("  - visibility_periods_used > 8")
        print("  - astrometric_excess_noise < 1.0")
        
        query = self.build_quality_query(max_sources)
        
        try:
            # Submit query to Gaia archive
            job = Gaia.launch_job_async(query, verbose=True)
            results = job.get_results()
            
            print(f"Downloaded {len(results):,} sources")
            
            # Convert to pandas DataFrame
            df = results.to_pandas()
            
            if save_raw:
                raw_file = os.path.join(self.output_dir, "gaia_dr3_raw.csv")
                df.to_csv(raw_file, index=False)
                print(f"Raw data saved to: {raw_file}")
            
            return df
            
        except Exception as e:
            print(f"Error downloading Gaia data: {e}")
            return None
    
    def add_derived_features(self, df):
        """
        Add derived features to the Gaia dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw Gaia data
            
        Returns:
        --------
        pandas.DataFrame : Data with derived features
        """
        print("Computing derived features...")
        
        # Distance from parallax (in parsecs)
        df['distance_pc'] = 1000.0 / df['parallax']  # parallax in mas
        
        # Absolute G magnitude
        df['abs_g_mag'] = df['phot_g_mean_mag'] - 5 * np.log10(df['distance_pc']) + 5
        
        # Total proper motion
        df['pm_total'] = np.sqrt(df['pmra']**2 + df['pmdec']**2)
        
        # Tangential velocity (km/s)
        # v_tan = 4.74 * mu * d, where mu is in mas/yr and d is in pc
        df['v_tan_km_s'] = 4.74 * df['pm_total'] * df['distance_pc'] / 1000.0
        
        # Color index (already exists as bp_rp, but ensure it's clean)
        df['color_bp_rp'] = df['bp_rp']
        
        # Reduced proper motion (proxy for luminosity class)
        df['reduced_pm'] = df['phot_g_mean_mag'] + 5 * np.log10(df['pm_total']) + 5
        
        print(f"Added derived features: distance_pc, abs_g_mag, pm_total, v_tan_km_s, reduced_pm")
        
        return df
    
    def save_processed_data(self, df, filename="gaia_dr3_processed.csv"):
        """Save processed data to file."""
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Processed data saved to: {filepath}")
        return filepath

def main():
    """Main function to run data acquisition."""
    print("=== Gaia DR3 Data Acquisition ===")
    
    # Initialize data acquisition
    gaia_acq = GaiaDataAcquisition()
    
    # Download data (start with smaller sample for testing)
    df = gaia_acq.download_gaia_data(max_sources=100000)  # 100k for initial testing
    
    if df is not None:
        print(f"\nDataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Add derived features
        df_processed = gaia_acq.add_derived_features(df)
        
        # Save processed data
        gaia_acq.save_processed_data(df_processed)
        
        print("\n=== Data Acquisition Complete ===")
        print(f"Final dataset: {df_processed.shape[0]:,} sources, {df_processed.shape[1]} features")
    else:
        print("Data acquisition failed!")

if __name__ == "__main__":
    main()
