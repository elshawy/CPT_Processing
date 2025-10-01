# -*- coding: utf-8 -*-
"""
=============================================================
Integrated CPT-Vs Data Processing and Layer Analysis Script
-------------------------------------------------------------
Purpose:
  This script merges CPT and Vs data files.
=============================================================
"""
import pandas as pd
import numpy as np
import os
import glob
import re
import sys
import ruptures as rpt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# =============================================================
# Vs Data Processing Toolkit
# =============================================================

# -------------------------------------------------------------
# Utility Function: Geometric Mean with Square Root Stabilization
# -------------------------------------------------------------
def geometric_mean_with_sqrt(data):
    """
    Calculates the geometric mean of a series of data points using
    a square root stabilization method to improve numerical stability.
    
    Args:
        data (array-like): The input data.
    
    Returns:
        float: The geometric mean, or None if the data is empty.
    """
    data = np.array(data)
    if len(data) == 0:
        return None
    # Apply square root, compute geometric mean, then square the result.
    sqrt_data = np.sqrt(data)
    gm_sqrt = np.exp(np.mean(np.log(sqrt_data)))
    return gm_sqrt ** 2

# -------------------------------------------------------------
# Interval Generation: Vs Depth Intervals
# -------------------------------------------------------------
def calculate_vs_intervals(depths):
    """
    Generates depth intervals for Vs profile analysis. The intervals are
    defined by the unique measured Vs depths up to a maximum depth of 30m.
    
    Args:
        depths (list): A list of depths from the measured Vs data.
        
    Returns:
        list of tuples: A list of (start_depth, end_depth) intervals.
    """
    depths = [min(d, 30) for d in depths]
    min_depth = 0
    depths = sorted(list(set(depths)))
    intervals = []
    if len(depths) > 0:
        intervals.append((min_depth, depths[0]))
        intervals.extend([(depths[i], depths[i + 1]) for i in range(len(depths) - 1)])
    return intervals

# -------------------------------------------------------------
# Geometric Mean Calculation: Estimated vs. Measured Vs
# Measured Vs selected based on the middle of the interval (depth_mid)
# -------------------------------------------------------------
def compute_geometric_means(vs_intervals, measured_vs, estimated_vs):
    """
    Computes the geometric mean for estimated Vs values
    within depth intervals from measured Vs profile.
    
    Args:
        vs_intervals (list of tuples): Depth intervals (start, end).
        measured_vs (list of tuples): Raw measured Vs data (depth, Vs).
        estimated_vs (pd.DataFrame): DataFrame with estimated Vs profiles.
        
    Returns:
        pd.DataFrame: A DataFrame containing the geometric mean results
                      for each interval.
    """
    results = []
    vs_cols = [c for c in estimated_vs.columns if c != 'Depth']

    # Convert measured_vs to a DataFrame for easier processing
    measured_df = pd.DataFrame(measured_vs, columns=['Depth', 'Vs']).sort_values('Depth')

    for start_depth, end_depth in vs_intervals:
        # Estimated Vs geometric mean
        interval_estimated_vs = estimated_vs[
            (estimated_vs['Depth'] >= start_depth) &
            (estimated_vs['Depth'] < end_depth)
        ]
        gm_values = [geometric_mean_with_sqrt(interval_estimated_vs[col].values)
                     if not interval_estimated_vs.empty else None
                     for col in vs_cols]

        # Measured Vs: find the closest value to the interval midpoint
        mid_depth = (start_depth + end_depth) / 2
        if not measured_df.empty:
            closest_idx = (measured_df['Depth'] - mid_depth).abs().idxmin()
            measured_value = measured_df.loc[closest_idx, 'Vs']
        else:
            measured_value = None

        results.append([start_depth, end_depth, measured_value] + gm_values)

    return pd.DataFrame(
        results,
        columns=["Start Depth", "End Depth", "Measured Vs"] + vs_cols
    )

# -------------------------------------------------------------
# Helper: Extract unique file key (numeric part)
# -------------------------------------------------------------
def extract_key(filename):
    """
    Extracts a unique numeric key from a filename for matching purposes.
    
    Args:
        filename (str): The name of the file.
        
    Returns:
        str: The extracted numeric key, or the original filename if no key is found.
    """
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else filename

# -------------------------------------------------------------
# Workflow 1: Process Geometric Means
# -------------------------------------------------------------
def process_geometric_mean(Vs_folder, Estimated_Vs_cpt_folder, output_folder, combined_output):
    """
    Executes a workflow to calculate geometric means for both estimated
    and measured Vs data, saving results as individual and combined CSVs.
    
    Args:
        Vs_folder (str): Path to the folder containing measured Vs data.
        Estimated_Vs_cpt_folder (str): Path to the folder containing estimated Vs data.
        output_folder (str): The root directory for all output files.
        combined_output (str): The path for the final combined CSV file.
    """
    indiv_folder = os.path.join(output_folder, "individual")
    os.makedirs(indiv_folder, exist_ok=True)

    all_results = []
    vs_files = {extract_key(os.path.basename(f)): f for f in glob.glob(os.path.join(Vs_folder, "*.csv"))}
    cpt_files = {extract_key(os.path.basename(f)): f for f in glob.glob(os.path.join(Estimated_Vs_cpt_folder, "*.csv"))}
    common_keys = set(vs_files.keys()) & set(cpt_files.keys())

    for key in common_keys:
        vs_df = pd.read_csv(vs_files[key])
        cpt_df = pd.read_csv(cpt_files[key]).drop(columns=['u'], errors='ignore')

        if 'd' in vs_df.columns and 'vs' in vs_df.columns and 'Depth' in cpt_df.columns:
            measured_vs = list(zip(vs_df['d'], vs_df['vs']))
            depths = vs_df['d'].values
            vs_intervals = calculate_vs_intervals(depths)
            vs_cols = [c for c in cpt_df.columns if c != 'Depth']
            result_df = compute_geometric_means(vs_intervals, measured_vs, cpt_df)
            new_columns = result_df.columns.tolist()
            for i, col in enumerate(new_columns):
                if col in vs_cols:
                    new_columns[i] = f'{col} (Geometric Mean)'
            result_df.columns = new_columns
            result_df = result_df.dropna(thresh=result_df.shape[1] - 3)

            # Save individual result
            result_df.to_csv(os.path.join(indiv_folder, f"Processed_{key}_geometric_mean.csv"), index=False)

            # Add File Name column for combined output
            result_df.insert(0, "File Name", os.path.basename(vs_files[key]))
            all_results.append(result_df)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df = final_df.dropna(thresh=final_df.shape[1] - 3)
        final_df.to_csv(combined_output, index=False)

    print(f"Geometric mean calculations complete. Results saved to '{output_folder}'.")


# =============================================================
# Layer Analysis and Plotting (for Mode 2)
# =============================================================

def geometric_mean(series):
    """
    Calculates the standard geometric mean of a pandas Series.
    
    Args:
        series (pd.Series): The input data.
        
    Returns:
        float: The geometric mean, or NaN if the series is empty.
    """
    series = series[series > 0]
    if series.empty:
        return np.nan
    return np.exp(np.log(series).mean())


def analyze_cpt_for_layers(file_path, output_csv_folder, output_plot_folder, min_thickness_m):
    """
    Analyzes CPT data to calculate the Ic index, detect layers using
    change-point detection, and saves both a processed CSV and a plot.
    
    Args:
        file_path (str): Path to the CPT data file.
        output_csv_folder (str): Directory to save the layer analysis CSV.
        output_plot_folder (str): Directory to save the layer plot image.
        min_thickness_m (float): Minimum layer thickness in meters.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    df_cols = [col.lower() for col in df.columns]

    if 'qc' in df_cols:
        qc_col = df.columns[df_cols.index('qc')]
        qc = df[qc_col].values
    else:
        print(f"Required column 'Qc' or 'qc' not found in {file_path}. Skipping.")
        return

    if 'fs' in df_cols:
        fs_col = df.columns[df_cols.index('fs')]
        fs = df[fs_col].values
    else:
        print(f"Required column 'Fs' or 'fs' not found in {file_path}. Using Fs = 0.")
        fs = np.zeros(len(df))

    if 'u' in df_cols:
        u_col = df.columns[df_cols.index('u')]
        u = df[u_col].values
    else:
        print(f"Required column 'U' or 'u' not found in {file_path}. Using U = 0.")
        u = np.zeros(len(df))

    depth = df['Depth'].values

    pa = 0.1
    ground_water_level = 1.0
    net_area_ratio = 0.8
    qt = qc + u * (1 - net_area_ratio)
    default_gamma = 0.00981 * 1.9
    gamma_w = 9.80665

    Rf_safe = np.where(qt <= 0, np.nan, (fs / qt) * 100)
    gamma = ((0.27 * np.log10(np.where(Rf_safe > 0, Rf_safe, np.nan)) + 0.36 * np.log10(np.where(qt / pa > 0, qt / pa, np.nan)) + 1.236) * gamma_w) / 1000
    gamma = np.where((qc <= 0) | (fs <= 0) | np.isnan(gamma), default_gamma, gamma)
    gamma = np.maximum(14.0 / 1000, gamma)

    totalStress = np.zeros(len(depth))
    u0 = np.zeros(len(depth))
    if len(depth) > 0:
        totalStress[0] = gamma[0] * depth[0]
        for i in range(1, len(depth)):
            totalStress[i] = gamma[i] * (depth[i] - depth[i - 1]) + totalStress[i - 1]
            if depth[i] >= ground_water_level:
                u0[i] = 0.00981 * (depth[i] - ground_water_level)

    effStress = totalStress - u0
    Fr_safe = np.where(qt - totalStress <= 0, np.nan, (fs / (qt - totalStress)) * 100)
    Ic = np.zeros(len(depth))
    n = 0.5 * np.ones(len(depth))

    max_iterations = 1000000
    for i in range(len(n)):
        deltan = 1
        iteration_counter = 0
        while deltan >= 0.01 and iteration_counter < max_iterations:
            n0 = n[i]
            if effStress[i] <= 0:
                cN = 1.7
            else:
                cN = (pa / effStress[i]) ** n[i]
            if cN > 1.7:
                cN = 1.7
            
            Qtn_val = ((qt[i] - totalStress[i]) / pa) * cN
            
            if Qtn_val <= 0 or Fr_safe[i] <= 0:
                Ic[i] = np.nan
                n[i] = np.nan
                break
            
            Ic[i] = ((3.47 - np.log10(Qtn_val))**2 + (np.log10(Fr_safe[i]) + 1.22)**2)**0.5
            n[i] = 0.381 * Ic[i] + 0.05 * (effStress[i] / pa) - 0.15
            if n[i] > 1:
                n[i] = 1
            deltan = np.abs(n0 - n[i])
            iteration_counter += 1
        
        if iteration_counter >= max_iterations:
            print(f"Warning: Exceeded max iterations for file {file_path} at depth {depth[i]:.2f}m. Calculation may not have converged.")

    temp_df = pd.DataFrame({'Depth': depth, 'Qc': qc, 'Ic': Ic})
    temp_df.dropna(subset=['Ic'], inplace=True)

    if temp_df.empty:
        print(f"Warning: All calculated Ic values for {file_path} are invalid. Skipping layer analysis.")
        return

    data_for_segmentation = StandardScaler().fit_transform(temp_df[['Qc', 'Ic']].values)
    depth_interval = np.median(np.diff(temp_df['Depth'].values))
    if depth_interval == 0 or np.isnan(depth_interval):
        print(f"Could not determine depth interval for {file_path}. Skipping min_thickness setting.")
        min_size = 0.5
    else:
        min_size = max(1, int(min_thickness_m / depth_interval))

    print(f"Using min_size of {min_size} data points (approx. {min_thickness_m} meters).")

    algo = rpt.Pelt(model="l2", jump=1, min_size=min_size).fit(data_for_segmentation)
    result = algo.predict(pen=5)
    
    if result and result[-1] == len(data_for_segmentation):
        result = result[:-1]

    layer_depths = temp_df.iloc[result]['Depth'].values
    layer_indices = sorted([df[df['Depth'] == d].index[0] for d in layer_depths])

    if 0 not in layer_indices:
        layer_indices.insert(0, 0)
    if len(df) not in layer_indices:
        layer_indices.append(len(df))

    file_name_prefix = os.path.splitext(os.path.basename(file_path))[0]
    fig, ax1 = plt.subplots(figsize=(8, 10))
    valid_indices = temp_df.index.tolist()
    ax1.plot(temp_df['Ic'].values, temp_df['Depth'].values, label='Ic', color='blue')
    ax1.set_xlabel('Ic (blue) & qc (red)')
    ax1.set_ylabel('Depth')
    ax1.invert_yaxis()
    ax2 = ax1.twiny()
    ax2.plot(temp_df['Qc'].values, temp_df['Depth'].values, label='Qc', color='red')
    ax1.set_title(f"qc & Ic with Layers Detected in {os.path.basename(file_path)}")
    for idx in layer_indices:
        if idx < len(df):
            depth_val = depth[idx]
            ax1.axhline(y=depth_val, color='r', linestyle='--', linewidth=1.5)

    plt.tight_layout()
    plot_path = os.path.join(output_plot_folder, f"{file_name_prefix}_layers_plot_qc_Ic.png")
    plt.savefig(plot_path)
    plt.close(fig)

    print(f"Layer plot saved to {plot_path}")

    stats_data = []
    df['Ic_calculated'] = Ic
    numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col not in ['Depth', 'Ic']]

    for i in range(len(layer_indices) - 1):
        start_idx = layer_indices[i]
        end_idx = layer_indices[i + 1]
        layer_df = df.iloc[start_idx:end_idx]
        stats = {
            'Layer': i + 1,
            'Start Depth': layer_df['Depth'].iloc[0] if not layer_df.empty else np.nan,
            'End Depth': layer_df['Depth'].iloc[-1] if not layer_df.empty else np.nan,
            'Thickness': (layer_df['Depth'].iloc[-1] - layer_df['Depth'].iloc[0]) if not layer_df.empty else np.nan
        }
        for col in numeric_cols:
            if col in layer_df.columns:
                stats[f'{col} (Geometric Mean)'] = geometric_mean(layer_df[col])
            else:
                stats[f'{col} (Geometric Mean)'] = np.nan

        stats_data.append(stats)
    
    stats_df = pd.DataFrame(stats_data)
    stats_file_path = os.path.join(output_csv_folder, f"{file_name_prefix}_layers_stats.csv")
    stats_df.to_csv(stats_file_path, index=False)

    print(f"Layer statistics saved to {stats_file_path}")
    return stats_df

# -------------------------------------------------------------
# Workflow 3: Merge All Measured Vs into CPT Profiles
# -------------------------------------------------------------
def merge_all_profiles(Estimated_Vs_cpt_folder, Vs_folder, output_folder):
    """
    Merges all measured Vs data into the corresponding CPT profiles based on
    a matching key, saving individual and combined results.
    
    Args:
        Estimated_Vs_cpt_folder (str): Path to the folder with CPT profiles.
        Vs_folder (str): Path to the folder with measured Vs data.
        output_folder (str): The root directory for all output files.
    """
    indiv_folder = os.path.join(output_folder, "individual")
    os.makedirs(indiv_folder, exist_ok=True)

    final_df = pd.DataFrame()
    cpt_files = glob.glob(os.path.join(Estimated_Vs_cpt_folder, '*.csv'))
    vs_dict = {extract_key(os.path.basename(f)): f for f in glob.glob(os.path.join(Vs_folder, '*.csv'))}

    for qc_file in cpt_files:
        fname = os.path.basename(qc_file)
        code = extract_key(fname)
        df_qc = pd.read_csv(qc_file)

        if code in vs_dict:
            df_vs = pd.read_csv(vs_dict[code]).sort_values('d').reset_index(drop=True)

            if df_vs.loc[0, 'd'] > 0:
                first_vs = df_vs.loc[0, 'vs']
                new_row = pd.DataFrame([{'d': 0, 'vs': first_vs}])
                df_vs = pd.concat([new_row, df_vs], ignore_index=True).sort_values('d').reset_index(drop=True)

            df_vs['d'] = df_vs['d'].astype(float)

            matched = pd.merge_asof(
                df_qc[['Depth']].sort_values('Depth'),
                df_vs,
                left_on='Depth',
                right_on='d',
                direction='backward'
            )
            matched = matched.set_index(df_qc.index)
            df_qc['Measure Vs'] = matched['vs']
        else:
            df_qc['Measure Vs'] = None

        # Save individual result
        out_path = os.path.join(indiv_folder, f"{code}_merged.csv")
        df_qc.to_csv(out_path, index=False)

        # Add File Name column for combined output
        df_qc.insert(0, "File Name", fname)
        final_df = pd.concat([final_df, df_qc], ignore_index=True)

    output_path = os.path.join(output_folder, 'Mode3_combined_results.csv')
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"All files have been merged and saved to: {output_path}")

# -------------------------------------------------------------
# Main Execution Block
# -------------------------------------------------------------
def run_mode2_workflow(Estimated_Vs_cpt_folder, Vs_folder, output_folder):
    """
    Executes the Vs-CPT data merging and layer analysis workflow.
    - Performs layer analysis first, then final merging.
    
    Args:
        Estimated_Vs_cpt_folder (str): Path to the folder with CPT data files.
        Vs_folder (str): Path to the folder with Vs data files.
        output_folder (str): The root directory for all output.
    """
    print("\n[Vs-CPT Data Merging and Layer Analysis Mode]")
    print("\n--- Current Execution Information ---")
    print(f"Input CPT Folder: {os.path.abspath(Estimated_Vs_cpt_folder)}")
    print(f"Input Vs Folder: {os.path.abspath(Vs_folder)}")

    if not os.path.isdir(Estimated_Vs_cpt_folder) or not os.path.isdir(Vs_folder):
        print("Error: The specified CPT or Vs folder does not exist.")
        print("Please check the paths. It is recommended to use quotation marks for the full path.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    main_output_dir = os.path.join(output_folder, f"Mode2_Analysis_Results_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)
    
    analysis_csv_folder = os.path.join(main_output_dir, "Layer_Analysis_CSVs")
    analysis_plot_folder = os.path.join(main_output_dir, "Layer_Analysis_Plots")
    merged_results_folder = os.path.join(main_output_dir, "Merged_Results")
    
    os.makedirs(analysis_csv_folder, exist_ok=True)
    os.makedirs(analysis_plot_folder, exist_ok=True)
    os.makedirs(merged_results_folder, exist_ok=True)

    print("\n=============================================================")
    print("Starting Vs & CPT Integrated Analysis.")
    print("=============================================================")
    print("\n--- Step 1: Analyzing CPT files for layers and generating statistics ---")
    
    cpt_files = glob.glob(os.path.join(Estimated_Vs_cpt_folder, '*.csv'))
    if not cpt_files:
        print("Warning: No CPT files found to analyze. Exiting program.")
        sys.exit(0)
        
    # Set minimum thickness
    min_thickness_m = 1
    for cpt_file in cpt_files:
        print(f"Processing file: {os.path.basename(cpt_file)}")
        analyze_cpt_for_layers(cpt_file, analysis_csv_folder, analysis_plot_folder, min_thickness_m)
        print("-" * 50)
    
    print("\n--- Step 1 Complete ---")
    print(f"Layer analysis result CSVs are in '{analysis_csv_folder}', and plots are in '{analysis_plot_folder}'.")

    print("\n--- Step 2: Merging analyzed CSV files with Vs data to create final results ---")
    
    final_df = pd.DataFrame()
    processed_cpt_files = glob.glob(os.path.join(analysis_csv_folder, '*.csv'))
    vs_dict = {extract_key(os.path.basename(f)): f for f in glob.glob(os.path.join(Vs_folder, '*.csv'))}
    
    for processed_file in processed_cpt_files:
        file_name = os.path.basename(processed_file)
        unique_code = extract_key(file_name)
        df_cpt_processed = pd.read_csv(processed_file)
        
        # FIX: Add 'File Name' column first to ensure it is preserved after merging.
        df_cpt_processed.insert(0, "File Name", file_name)
        
        if 'Start Depth' not in df_cpt_processed.columns or 'End Depth' not in df_cpt_processed.columns:
            print(f"Warning: The file {file_name} is missing 'Start Depth' or 'End Depth' columns. Skipping merge.")
            continue
        
        if unique_code in vs_dict:
            df_vs = pd.read_csv(vs_dict[unique_code]).sort_values('d').reset_index(drop=True)
            df_vs.rename(columns={'d': 'Depth_vs', 'vs': 'Vs_Measured_Raw'}, inplace=True)
            df_vs['Measured Vs'] = df_vs['Depth_vs'].astype(float)
            
            # Vs data is at 1m intervals, so match to the midpoint of the layer interval.
            df_cpt_processed['Depth_mid'] = (df_cpt_processed['Start Depth'] + df_cpt_processed['End Depth']) / 2
            
            # --- FIX: Drop rows with NaN in the merge key before merging
            df_cpt_processed = df_cpt_processed.dropna(subset=['Depth_mid']).copy()
            
            # Merge the closest Vs_Measured_Raw value at the nearest depth
            merged_df = pd.merge_asof(
                df_cpt_processed.sort_values('Depth_mid'),
                df_vs.sort_values('Measured Vs'),
                left_on='Depth_mid',
                right_on='Measured Vs',
                direction='nearest'
            )
            
            # Rename and reorder columns
            merged_df.rename(columns={'Depth_mid': 'Depth', 'Vs_Measured_Raw': 'Measured Vs'}, inplace=True)
            cols = list(merged_df.columns)
            
            # Reorder columns to the desired order (Depth and Measured Vs after End Depth)
            reordered_cols = ['File Name', 'Layer', 'Start Depth', 'End Depth']
            for col in ['Depth', 'Measured Vs']:
                if col in cols:
                    reordered_cols.append(col)
            
            for col in cols:
                if col not in reordered_cols:
                    reordered_cols.append(col)
            
            merged_df = merged_df[reordered_cols]
            
            # Save individual merged result CSV to a separate folder
            individual_merged_path = os.path.join(merged_results_folder, f"Processed_{unique_code}_merged_layers.csv")
            merged_df.to_csv(individual_merged_path, index=False, encoding='utf-8-sig')
            print(f"Individual merged result saved: '{os.path.basename(individual_merged_path)}'")
            
            #merged_df.insert(0, "File Name", file_name)  # This code is no longer necessary.
            final_df = pd.concat([final_df, merged_df], ignore_index=True)
        else:
            print(f"Warning: Vs data file for {file_name} does not exist. Cannot merge Vs measurements.")
            # In this branch, 'File Name' is already added to df_cpt_processed, so no separate handling is needed.
            final_df = pd.concat([final_df, df_cpt_processed], ignore_index=True)
    
    if not final_df.empty:
        output_path = os.path.join(main_output_dir, 'Mode2_combined_results.csv')
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Final combined results have been successfully saved to '{output_path}'.")
    else:
        print("Warning: No files to merge. Final results were not generated.")

    print("\n--- Step 2 Complete ---")
    print("\nAll analysis and merging are complete. Result files are organized in the output folder.")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python 3.CPT_Vs_Processing.py <mode:1|2|3> <Estimated_Vs_cpt_folder> <Vs_folder> <output_folder>")
        print("  <Mode> 1: Merge after calculating geometric mean based on measured Vs")
        print("  <Mode> 2: Merge CPT-Vs data after performing layer analysis based on qc & Ic")
        print("  <Mode> 3: Merge all values")
        sys.exit(1)

    mode = sys.argv[1]
    Estimated_Vs_cpt_folder = sys.argv[2]
    Vs_folder = sys.argv[3]
    output_folder = sys.argv[4]

    # Create output folder
    try:
        os.makedirs(output_folder, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output folder '{output_folder}'. {e}")
        sys.exit(1)

    if not os.path.isdir(Estimated_Vs_cpt_folder) or not os.path.isdir(Vs_folder):
        print("Error: One or both input folders do not exist.")
        sys.exit(1)

    if mode == '1':
        print("\n[Geometric Mean Calculation and Vs Data Comparison Mode]")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        main_output_dir = os.path.join(output_folder, f"Mode1_Analysis_Results_{timestamp}")
        os.makedirs(main_output_dir, exist_ok=True)
        combined_output = os.path.join(main_output_dir, "Mode1_combined_results.csv")
        process_geometric_mean(Vs_folder, Estimated_Vs_cpt_folder, main_output_dir, combined_output)

    elif mode == '2':
        run_mode2_workflow(Estimated_Vs_cpt_folder, Vs_folder, output_folder)
        
    elif mode == '3':
        print("\n[Merging All Profiles Mode]")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        main_output_dir = os.path.join(output_folder, f"Mode3_Analysis_Results_{timestamp}")
        os.makedirs(main_output_dir, exist_ok=True)
        merge_all_profiles(Estimated_Vs_cpt_folder, Vs_folder, main_output_dir)

    else:
        print("Invalid mode. Please select 1, 2, or 3.")
        sys.exit(1)








