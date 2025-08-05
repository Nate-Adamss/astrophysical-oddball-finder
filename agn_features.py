"""
Feature engineering for AGN classification using Gaia + Pan-STARRS data.

This module implements physically meaningful derived features that can help distinguish AGNs from other astrophysical sources.
"""

import numpy as np

def calculate_reduced_proper_motion(parallax, pmra, pmdec, g_mag):
    """
    Calculate reduced proper motion (RPM) in the G-band.
    RPM is a powerful tool for separating AGNs from stars and galaxies.
    
    Parameters:
    -----------
    parallax : float
        Parallax in milliarcseconds
    pmra : float
        Proper motion in right ascension (mas/yr)
    pmdec : float
        Proper motion in declination (mas/yr)
    g_mag : float
        Gaia G-band magnitude
        
    Returns:
    --------
    float
        Reduced proper motion in the G-band
    """
    # Calculate total proper motion
    pm_total = np.sqrt(pmra**2 + pmdec**2)
    # Convert parallax to distance (kpc)
    distance = 1.0 / (parallax * 1e-3)  # Convert mas to kpc
    # Calculate reduced proper motion
    rpm = g_mag + 5 * np.log10(distance) - 5
    return rpm

def calculate_color_indices(bp_rp, ps_i, ps_z):
    """
    Calculate various color indices that can help distinguish AGNs.
    
    Parameters:
    -----------
    bp_rp : float
        Gaia BP-RP color
    ps_i : float
        Pan-STARRS i-band magnitude
    ps_z : float
        Pan-STARRS z-band magnitude
        
    Returns:
    --------
    dict
        Dictionary containing various color indices
    """
    indices = {}
    
    # Gaia color indices
    indices['bp_rp'] = bp_rp
    
    # Pan-STARRS color indices
    indices['i_z'] = ps_i - ps_z
    
    # Combined color indices
    indices['bp_rp_i_z'] = bp_rp * (ps_i - ps_z)
    
    return indices

def calculate_variability_features(variability_indicators):
    """
    Calculate variability features that can help identify AGNs.
    
    Parameters:
    -----------
    variability_indicators : dict
        Dictionary containing variability indicators from Gaia
        
    Returns:
    --------
    dict
        Dictionary containing variability features
    """
    features = {}
    
    # Calculate variability amplitude
    features['variability_amplitude'] = variability_indicators.get('amplitude', 0)
    
    # Calculate variability period
    features['variability_period'] = variability_indicators.get('period', 0)
    
    # Calculate variability index (e.g., excess noise)
    features['variability_index'] = variability_indicators.get('excess_noise', 0)
    
    return features

def calculate_agn_features(gaia_data, panstarrs_data, variability_data):
    """
    Main function to calculate all AGN classification features.
    
    Parameters:
    -----------
    gaia_data : dict
        Dictionary containing Gaia data (parallax, pmra, pmdec, G_mag, BP-RP)
    panstarrs_data : dict
        Dictionary containing Pan-STARRS data (i, z magnitudes)
    variability_data : dict
        Dictionary containing variability indicators
        
    Returns:
    --------
    dict
        Dictionary containing all calculated features
    """
    features = {}
    
    # Calculate reduced proper motion
    features['rpm'] = calculate_reduced_proper_motion(
        gaia_data['parallax'],
        gaia_data['pmra'],
        gaia_data['pmdec'],
        gaia_data['g_mag']
    )
    
    # Calculate color indices
    color_indices = calculate_color_indices(
        gaia_data['bp_rp'],
        panstarrs_data['i_mag'],
        panstarrs_data['z_mag']
    )
    features.update(color_indices)
    
    # Calculate variability features
    variability_features = calculate_variability_features(variability_data)
    features.update(variability_features)
    
    # Calculate additional features
    features['distance'] = 1.0 / (gaia_data['parallax'] * 1e-3)  # Distance in kpc
    features['pm_total'] = np.sqrt(gaia_data['pmra']**2 + gaia_data['pmdec']**2)  # Total proper motion
    
    return features
