# -*- coding: utf-8 -*-
"""
=============================================================
Integrated CPT-Vs Correlation and Data Processing Script
-------------------------------------------------------------
Purpose:
  This script reads CPT (Cone Penetration Test) data files,
  applies selected Vs correlation models to generate Vs depth profiles.
  It then processes the generated Vs data based on user input and
  merges it with the original data for a comprehensive output.
=============================================================
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os
import re
import numpy as np
import glob
from vs_calc import CPT, VsProfile, vs30_correlations

# -------------------------------------------------------------
# CPT-Vs Correlation Models Definition
# -------------------------------------------------------------
CPT_CORRELATIONS = {
    "andrus_2007_holocene": "andrus_2007_holocene",
    "andrus_2007_pleistocene": "andrus_2007_pleistocene",
    "andrus_2007_tertiary_age_cooper_marl": "andrus_2007_tertiary_age_cooper_marl",
    "robertson_2009": "robertson_2009",
    "hegazy_2006": "hegazy_2006",
    "mcgann_2015": "mcgann_2015",
    "mcgann_2018": "mcgann_2018",
}

# -------------------------------------------------------------
# User Input: Folder and Model Selection
# -------------------------------------------------------------
print("=============================================================")
print(" Vs Profile Calculator and Integration Script ")
print("=============================================================")

# Prompt user for the input folder path containing CPT data files (CSV format).
input_folder_str = input("Enter the folder path containing the CPT data: ").strip()
examples_dir = Path(input_folder_str)

if not examples_dir.exists():
    print(f"Error: The specified folder '{examples_dir}' does not exist. Exiting script.")
    sys.exit(1)

print("\nAvailable Vs Correlation Models:")
model_keys = list(CPT_CORRELATIONS.keys())
for i, model in enumerate(model_keys):
    print(f"  {i+1}. {model}")

# Prompt user to select one or more correlation models.
selected_models_str = input("Enter the model number(s) to use, separated by commas (e.g., 1,3,4): ").strip()

if selected_models_str == '8':
    selected_model_names = model_keys
    print("All available models have been selected.")
else:
    try:
        selected_model_indices = [int(i.strip()) - 1 for i in selected_models_str.split(',')]
        selected_model_names = [model_keys[i] for i in selected_model_indices if 0 <= i < len(model_keys)]
    except (ValueError, IndexError):
        print("Error: Invalid model number(s) provided. Exiting script.")
        sys.exit(1)

if not selected_model_names:
    print("Error: No valid models were selected. Exiting script.")
    sys.exit(1)

# Prompt user for data cleaning option.
remove_nan = input("Do you want to remove rows with NaN or empty values? (Y/N): ").strip().upper() == "Y"

# -------------------------------------------------------------
# Define Output Directory
# -------------------------------------------------------------
# Create a unique output directory based on the current timestamp to avoid overwriting files.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path(f'./EstimatedVs_{timestamp}')
out_dir.mkdir(exist_ok=True)
log_file = out_dir / f"error_log_{timestamp}.txt"

print("\n-------------------------------------------------------------")
print(" Calculating Vs Profiles...")
print("-------------------------------------------------------------")

# -------------------------------------------------------------
# Main Processing Loop
# -------------------------------------------------------------
# Iterate through all CSV files in the user-specified input directory.
for file_path in examples_dir.glob('*.csv'):
    file_name = file_path.stem
    try:
        # Step 1: Load CPT Data
        # The vs_calc library requires a specific CPT file format.
        cpt = CPT.from_file(str(file_path))

        # Step 2: Dynamically Generate Vs Profiles and Store as DataFrames
        # Generate Vs profiles for each selected correlation model.
        vs_dfs = []
        for model_name in selected_model_names:
            # Use the VsProfile class to calculate Vs values from CPT data.
            vs_profile = VsProfile.from_cpt(cpt, CPT_CORRELATIONS[model_name])
            vs_df_temp = pd.DataFrame({
                'Depth': vs_profile.depth,
                f'{model_name} Vs': vs_profile.vs
            })
            vs_dfs.append(vs_df_temp)
        
        # Step 3: Prepare Input DataFrame
        # Read the original CSV file into a DataFrame for merging.
        df = pd.read_csv(file_path)
        if 'Depth' not in df.columns:
            print(f"Warning: The file {file_name} does not contain a 'Depth' column. Skipping this file.")
            continue
        # Round the depth column to ensure consistent merging.
        df.iloc[:, 0] = df.iloc[:, 0].round(2)

        # Step 4: Merge Vs Correlation Results with the main DataFrame
        if vs_dfs:
            # Merge the generated Vs profile data into the original DataFrame.
            # Start with the first model's DataFrame and iteratively merge the rest.
            vs_merged_df = vs_dfs[0]
            for i in range(1, len(vs_dfs)):
                vs_merged_df = pd.merge(vs_merged_df, vs_dfs[i], how='outer', on='Depth')
            df = pd.merge(df, vs_merged_df, how='left', on='Depth')

        # Step 5: Merge Ic Values
        # Add the Ic (soil behavior type index) values to the final DataFrame.
        ic_df = pd.DataFrame({'Depth': cpt.depth, 'Ic': cpt.Ic})
        # Use merge_asof for a nearest-match merge based on depth.
        df = pd.merge_asof(df.sort_values('Depth'),
                            ic_df.sort_values('Depth'),
                            on='Depth')
        df.iloc[:, 0] = df.iloc[:, 0].round(2)

        # Step 6: Optionally remove rows with NaN/empty values
        # Clean the data by dropping rows with any missing values, if requested by the user.
        if remove_nan:
            df = df.dropna(how="any")
            df = df[~df.eq("").any(axis=1)]

        # Step 7: Save Results
        # Write the final merged DataFrame to a new CSV file in the output directory.
        df.to_csv(out_dir / f'{file_name}_result.csv', index=False)
        print(f'Processing complete: {file_name}')

    except Exception as e:
        # Error Handling: Log any errors encountered during processing.
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(f"Error occurred while processing file {file_name}: {str(e)}\n")
        print(f"Skipping {file_name} due to an error.")

print("\n-------------------------------------------------------------")
print(" All Vs profiles have been successfully generated. ")
print("-------------------------------------------------------------")
print(f"The result files are saved in the '{out_dir}' folder.")
