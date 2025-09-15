# -*- coding: utf-8 -*-
"""
=============================================================
Vs Profile Comparison and Plotting Script
-------------------------------------------------------------
Purpose:
  This script compares and plots shear wave velocity (Vs)
  profiles from three different data sources:
  1. CPT-estimated Vs (from original CPT data)
  2. Processed CPT-estimated Vs (processed data with corrections)
  3. Measured Vs (field-measured Vs data)

Usage:
  Run the script from the command line with the three folder paths
  as arguments. Use 'N' to skip a folder if the data source is
  not available.

  python your_script_name.py <Processed_Vs_Folder> <Measured_Vs_Folder> <CPT_Vs_Folder>
  
  Example:
  python comparison.py ./results_Qc_Ic_1 N ./cpt_data
=============================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
from datetime import datetime

# Check for correct number of command-line arguments
if len(sys.argv) != 4:
    print("Usage: python your_script_name.py <Processed_Vs_Folder> <Measured_Vs_Folder> <CPT_Vs_Folder>")
    print("Example: python comparison.py ./results_Qc_Ic_1 ./vsmo ./cpt_data")
    print("Use 'N' to skip a folder.")
    sys.exit(1)

# Define folders from command-line arguments
processed_vs_folder = sys.argv[1] if sys.argv[1].upper() != 'N' else None
measured_vs_folder = sys.argv[2] if sys.argv[2].upper() != 'N' else None
cpt_vs_folder = sys.argv[3] if sys.argv[3].upper() != 'N' else None

# Create the output folder with a timestamp to prevent overwriting
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f'./comparison_plots_output_{timestamp}'
os.makedirs(output_folder, exist_ok=True)


def extract_file_key(filename):
    """
    Extracts a consistent file key based on the number of underscores in the filename.
    If there are at least two underscores, the key is the part between the first and second.
    If there is only one underscore, the key is the part after the underscore and before the extension.
    
    e.g., 'Processed_01_merged_layers.csv' -> '01'
    e.g., 'Measured_01.csv' -> '01'
    e.g., 'CPT_01.csv' -> '01'
    """
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) >= 3:
        return parts[1]
    elif len(parts) == 2:
        return parts[1].replace('.csv', '')
    return None


def get_file_map(folder):
    """
    Creates a dictionary mapping file keys to file paths for a given folder.
    This function uses the unified extract_file_key function.
    """
    file_map = {}
    if folder and os.path.exists(folder):
        for f in os.listdir(folder):
            if f.endswith('.csv'):
                code = extract_file_key(f)
                if code:
                    file_map[code] = os.path.join(folder, f)
    return file_map


# Get file maps for each specified folder using the unified key extractor
processed_file_map = get_file_map(processed_vs_folder)
measured_file_map = get_file_map(measured_vs_folder)
cpt_file_map = get_file_map(cpt_vs_folder)

# Find all unique file codes to iterate through
processed_codes = sorted(list(processed_file_map.keys()))

# Define the columns to be plotted for each data source
cpt_vs_cols = ['andrus_2007_holocene Vs', 'andrus_2007_pleistocene Vs', 'andrus_2007_tertiary_age_cooper_marl Vs', 'robertson_2009 Vs', 'hegazy_2006 Vs', 'mcgann_2015 Vs', 'mcgann_2018 Vs']
processed_vs_cols = ['andrus_2007_holocene Vs (Geometric Mean)', 'andrus_2007_pleistocene Vs (Geometric Mean)', 'andrus_2007_tertiary_age_cooper_marl Vs (Geometric Mean)', 'robertson_2009 Vs (Geometric Mean)', 'hegazy_2006 Vs (Geometric Mean)', 'mcgann_2015 Vs (Geometric Mean)', 'mcgann_2018 Vs (Geometric Mean)']

for code in processed_codes:
    print(f"Processing code: {code}")
    
    df_processed = None
    df_measured = None
    df_cpt = None

    try:
        # Load dataframes if paths exist
        if code in processed_file_map:
            df_processed = pd.read_csv(processed_file_map[code])
        if code in measured_file_map:
            df_measured = pd.read_csv(measured_file_map[code])
        if code in cpt_file_map:
            df_cpt = pd.read_csv(cpt_file_map[code])

    except FileNotFoundError as e:
        print(f"Skipping code '{code}' due to a missing file: {e}")
        continue
    except Exception as e:
        print(f"An error occurred while reading files for code '{code}': {e}")
        continue
    
    # Calculate the overall maximum Vs value for this specific code for consistent plotting
    max_vs_for_code = 0
    if df_cpt is not None:
        for col in cpt_vs_cols:
            if col in df_cpt.columns:
                max_vs_for_code = max(max_vs_for_code, df_cpt[col].max())

    if df_processed is not None:
        for col in processed_vs_cols:
            if col in df_processed.columns:
                max_vs_for_code = max(max_vs_for_code, df_processed[col].max())

    if df_measured is not None:
        if 'vs' in df_measured.columns:
            max_vs_for_code = max(max_vs_for_code, df_measured['vs'].max())

    # Add a buffer for better visualization
    if max_vs_for_code > 0:
        max_vs_for_code *= 1.1
    else:
        max_vs_for_code = 500  # Default to 500 if all Vs values are 0 or less

    # Create a subfolder for the current code's plots
    code_output_folder = os.path.join(output_folder, code)
    os.makedirs(code_output_folder, exist_ok=True)
    
    # Generate plots for each Vs correlation model
    for i, (col_cpt, col_processed) in enumerate(zip(cpt_vs_cols, processed_vs_cols)):
        plt.figure(figsize=(6, 8))
        plt.title(f'Vs Profile Comparison for {code} - {col_cpt}')
        plt.xlabel('Vs (m/s)')
        plt.ylabel('Depth (m)')
        plt.gca().invert_yaxis()

        # Plot CPT-estimated Vs (blue)
        if df_cpt is not None and col_cpt in df_cpt.columns and 'Depth' in df_cpt.columns:
            plt.plot(df_cpt[col_cpt], df_cpt['Depth'], label='CPT-estimated Vs', color='blue', linewidth=2)

        # Plot Processed CPT-estimated Vs (red, step)
        if df_processed is not None and col_processed in df_processed.columns:
            # Calculate start depth for each layer for step plot
            end_depths = df_processed['End Depth'].values
            start_depths = np.insert(end_depths[:-1], 0, 0)
            vs_vals = df_processed[col_processed].values

            y_coords_processed = np.empty(2 * len(start_depths))
            y_coords_processed[0::2] = start_depths
            y_coords_processed[1::2] = end_depths
            x_coords_processed = np.repeat(vs_vals, 2)
            plt.step(x_coords_processed, y_coords_processed, where='post', label='Processed CPT-estimated Vs', color='red', linestyle='-', linewidth=2)

        # Plot Measured Vs (green, step)
        if df_measured is not None and 'd' in df_measured.columns and 'vs' in df_measured.columns:
            # Prepare data for step plot of measured Vs
            vs2_depths = df_measured['d'].values
            vs2_vs_vals = df_measured['vs'].values

            vs2_start_depths = np.insert(vs2_depths[:-1], 0, 0)
            vs2_end_depths = vs2_depths

            y_coords_vs2 = np.empty(2 * len(vs2_start_depths))
            y_coords_vs2[0::2] = vs2_start_depths
            y_coords_vs2[1::2] = vs2_end_depths
            x_coords_vs2 = np.repeat(vs2_vs_vals, 2)

            plt.step(x_coords_vs2, y_coords_vs2, where='post', label='Measured Vs', color='green', linestyle='--', linewidth=2)

        # Set the x-axis limit for consistent visualization across plots
        plt.xlim(0, max_vs_for_code)

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot with a descriptive filename
        plot_filename = f'{code}_{col_cpt.replace(" ", "_")}.png'
        plot_path = os.path.join(code_output_folder, plot_filename)
        plt.savefig(plot_path)
        plt.close()

    print(f"Generated plots for code '{code}' and saved them to {code_output_folder}")

print("Script finished. All plots have been generated and saved.")
