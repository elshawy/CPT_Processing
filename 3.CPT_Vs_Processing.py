# -*- coding: utf-8 -*-
"""
=============================================================
Integrated CPT-Vs Data Processing and Layer Analysis Script
-------------------------------------------------------------
Purpose:
  This script merges CPT and Vs data files, supporting step-profile (staircase) matching
  for measured Vs in all modes.
  
  CRITICAL UPDATE: All modes (1, 2, 3) now filter CPT data to the maximum depth 
  of the measured Vs data for consistency in CPT-Vs comparison.
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

# -------------------------------------------------------------
# Utility Function: Geometric Mean with Square Root Stabilization
# -------------------------------------------------------------
def geometric_mean_with_sqrt(data):
    data = np.array(data)
    if len(data) == 0:
        return None
    # Filter out non-positive values before taking log
    positive_data = data[data > 0]
    if len(positive_data) == 0:
        return None
        
    sqrt_data = np.sqrt(positive_data)
    gm_sqrt = np.exp(np.mean(np.log(sqrt_data)))
    return gm_sqrt ** 2

# -------------------------------------------------------------
# Interval Generation: Vs Depth Intervals
# -------------------------------------------------------------
def calculate_vs_intervals(depths):
    # depths = [min(d, 30) for d in depths] # Removed depth limit as Vs data should already be filtered
    min_depth = 0
    # Ensure depths are unique and sorted
    depths = sorted(list(set(depths)))
    intervals = []
    
    if len(depths) > 0 and depths[0] > 0:
        intervals.append((min_depth, depths[0]))
        
    intervals.extend([(depths[i], depths[i + 1]) for i in range(len(depths) - 1)])
    
    # If the last depth is not 0, and there's a previous interval, close the last interval
    if len(depths) > 0:
        if len(intervals) == 0:
             # Only one depth point > 0, interval is (0, d)
            intervals.append((min_depth, depths[0]))
        else:
            # Check if the last interval is correctly defined up to the last point
            last_interval_end = intervals[-1][1]
            if last_interval_end < depths[-1]:
                 intervals.append((last_interval_end, depths[-1]))

    return intervals

# -------------------------------------------------------------
# Helper: Extract unique file key (numeric part)
# -------------------------------------------------------------
def extract_key(filename):
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else filename

# -------------------------------------------------------------
# Step Profile Utility: Get Vs assigner function for any depth
# -------------------------------------------------------------
# -------------------------------------------------------------
# Step Profile Utility: Get Vs assigner function for any depth
# (Modified to assign Vs_at_di to the interval [d_{i-1}, d_i))
# -------------------------------------------------------------
#def get_vs_step_assigner(df_vs):
#
#    df_vs = df_vs.sort_values('d').reset_index(drop=True)
#    
#    if not df_vs.empty and df_vs.iloc[0]['d'] > 0:
#        first_vs = df_vs.iloc[0]['vs']
#        new_row = pd.DataFrame([{'d': 0, 'vs': first_vs}])
#        df_vs = pd.concat([new_row, df_vs], ignore_index=True).sort_values('d').reset_index(drop=True)
#    
#    
#    df_vs['d_upper'] = df_vs['d']
#    
#    df_vs['d_lower'] = df_vs['d'].shift(1)
#    
#    df_vs = df_vs.astype({'d_lower': float, 'd_upper': float})
#    
#    df_vs.loc[0, 'd_lower'] = 0.0
#    
#    last_idx = df_vs.index[-1]
#    
#    last_d = df_vs.loc[last_idx, 'd']
#
#    df_vs.loc[last_idx, 'd_upper'] = last_d + 1e-4 
#    # -------------------------------------------------------------
#    
#    
#    def assign_vs(depth):
#        row = df_vs[(df_vs['d_lower'] <= depth) & (depth < df_vs['d_upper'])]
#        return row['vs'].iloc[0] if not row.empty else None
#    
#    return assign_vs
def get_vs_step_assigner(df_vs: pd.DataFrame):
    if df_vs.empty or 'd' not in df_vs.columns or 'vs' not in df_vs.columns:
        return lambda depth: np.nan

    df_vs = df_vs.sort_values('d').reset_index(drop=True)

    if df_vs.iloc[0]['d'] > 0:
        first_vs = df_vs.iloc[0]['vs']
        if len(df_vs) > 1:
            depth_diffs = np.diff(df_vs['d'].values)
            median_diff = np.median(depth_diffs)
        else:
            median_diff = df_vs.iloc[0]['d']
        new_depth = max(df_vs.iloc[0]['d'] - median_diff, 0.0)
        new_row = pd.DataFrame([{'d': new_depth, 'vs': first_vs}])
        df_vs = pd.concat([new_row, df_vs], ignore_index=True).sort_values('d').reset_index(drop=True)

    df_vs['d_upper'] = df_vs['d']
    df_vs['d_lower'] = df_vs['d'].shift(1)
    df_vs = df_vs.astype({'d_lower': float, 'd_upper': float})

    df_vs.loc[0, 'd_lower'] = df_vs.iloc[0]['d']

    last_idx = df_vs.index[-1]
    last_d = df_vs.loc[last_idx, 'd']
    df_vs.loc[last_idx, 'd_upper'] = last_d + 1e-4

    d_lower_arr = df_vs['d_lower'].values
    d_upper_arr = df_vs['d_upper'].values
    vs_arr = df_vs['vs'].values

    def assign_vs(depth):
        match = (d_lower_arr <= depth) & (depth < d_upper_arr)
        return vs_arr[match][0] if np.any(match) else None

    return assign_vs
# -------------------------------------------------------------
# Mode 1: Geometric Mean Calculation with Step Profile Matching
# -------------------------------------------------------------
def compute_geometric_means_step(vs_intervals, measured_vs, estimated_vs):
    """
    Computes geometric mean for estimated Vs, and assigns measured Vs step-wise per interval.
    """
    results = []
    vs_cols = [c for c in estimated_vs.columns if c != 'Depth']
    df_vs = pd.DataFrame(measured_vs, columns=['d', 'vs']).sort_values('d').reset_index(drop=True)
    assign_vs = get_vs_step_assigner(df_vs)
    
    for start_depth, end_depth in vs_intervals:
        # Check if the interval is valid
        if start_depth >= end_depth:
            continue
            
        mid_depth = (start_depth + end_depth) / 2  # Use mid-depth for correct step profile
        
        # Filter estimated Vs data for the current interval
        interval_estimated_vs = estimated_vs[
            (estimated_vs['Depth'] >= start_depth) & (estimated_vs['Depth'] < end_depth)
        ]
        
        # Calculate Geometric Mean for each estimated Vs column in the interval
        gm_values = [geometric_mean_with_sqrt(interval_estimated_vs[col].values)
                     if not interval_estimated_vs.empty else None
                     for col in vs_cols]
        
        # Assign measured Vs from the step profile at the mid-depth
        measured_value = assign_vs(mid_depth)
        
        results.append([start_depth, end_depth, mid_depth, measured_value] + gm_values)
        
    return pd.DataFrame(
        results,
        columns=["Start Depth", "End Depth", "Depth", "Measured Vs"] + vs_cols
    )

def process_geometric_mean(Vs_folder, Estimated_Vs_cpt_folder, output_folder, combined_output):
    indiv_folder = os.path.join(output_folder, "individual")
    os.makedirs(indiv_folder, exist_ok=True)
    all_results = []
    vs_files = {extract_key(os.path.basename(f)): f for f in glob.glob(os.path.join(Vs_folder, "*.csv"))}
    cpt_files = {extract_key(os.path.basename(f)): f for f in glob.glob(os.path.join(Estimated_Vs_cpt_folder, "*.csv"))}
    common_keys = set(vs_files.keys()) & set(cpt_files.keys())
    
    if not common_keys:
        print("Warning: No matching CPT and Vs files found based on their numeric key. Exiting Mode 1.")
        return

    for key in common_keys:
        vs_df = pd.read_csv(vs_files[key])
        cpt_df = pd.read_csv(cpt_files[key]).drop(columns=['u'], errors='ignore')

        if 'd' in vs_df.columns and 'vs' in vs_df.columns and 'Depth' in cpt_df.columns:
            
            # =============================================================
            # CRITICAL ADDITION: Filter CPT data to max Vs depth (Mode 1)
            # =============================================================
            max_vs_depth = vs_df['d'].max()
            initial_cpt_rows = len(cpt_df)
            cpt_df_filtered = cpt_df[cpt_df['Depth'] <= max_vs_depth].copy()
            
            if len(cpt_df_filtered) < initial_cpt_rows:
                print(f"Info (Mode 1): {initial_cpt_rows - len(cpt_df_filtered)} CPT rows exceeding max Vs depth ({max_vs_depth:.2f}m) were removed from CPT file with key {key}.")
                
            if cpt_df_filtered.empty:
                print(f"Warning (Mode 1): CPT data for key {key} is empty after Vs depth filtering. Skipping.")
                continue
            # =============================================================
            
            measured_vs = list(zip(vs_df['d'], vs_df['vs']))
            depths = vs_df['d'].values
            vs_intervals = calculate_vs_intervals(depths)
            
            vs_cols = [c for c in cpt_df_filtered.columns if c != 'Depth']
            
            # Use the filtered CPT data for geometric mean calculation
            result_df = compute_geometric_means_step(vs_intervals, measured_vs, cpt_df_filtered)
            
            new_columns = result_df.columns.tolist()
            for i, col in enumerate(new_columns):
                if col in vs_cols:
                    new_columns[i] = f'{col} (Geometric Mean)'
            result_df.columns = new_columns
            
            # Drop rows where there are too many NaN values (e.g., more than 4 non-NaN values are required)
            result_df = result_df.dropna(thresh=result_df.shape[1] - 4) 
            
            if result_df.empty:
                print(f"Warning (Mode 1): Result data for key {key} is empty after dropping NaNs. Skipping.")
                continue

            result_df.to_csv(os.path.join(indiv_folder, f"Processed_{key}_geometric_mean.csv"), index=False)
            result_df.insert(0, "File Name", os.path.basename(vs_files[key]))
            all_results.append(result_df)
            
        else:
            print(f"Warning (Mode 1): Missing required columns ('d', 'vs' in Vs file or 'Depth' in CPT file) for key {key}. Skipping.")
            
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        # Final cleanup for the combined file
        final_df = final_df.dropna(thresh=final_df.shape[1] - 4)
        final_df.to_csv(combined_output, index=False)
    print(f"Geometric mean calculations complete. Results saved to '{output_folder}'.")

# -------------------------------------------------------------
# Standard Geometric Mean for Layer Statistics
# -------------------------------------------------------------
def geometric_mean(series):
    series = series[series > 0]
    if series.empty:
        return np.nan
    return np.exp(np.log(series).mean())

# -------------------------------------------------------------
# CPT Layer Analysis with Change-point Detection (Accepts DataFrame)
# -------------------------------------------------------------
def analyze_cpt_for_layers_df(df_cpt, file_name, output_csv_folder, output_plot_folder, min_thickness_m):
    
    if df_cpt.empty:
        print(f"Warning: Input DataFrame for {file_name} is empty. Skipping layer analysis.")
        return None

    df_cols = [col.lower() for col in df_cpt.columns]
    
    # Standardize column names for processing
    col_map = {}
    if 'depth' in df_cols:
        col_map['Depth'] = df_cpt.columns[df_cols.index('depth')]
    
    # Check for required columns and handle missing ones
    if 'qc' in df_cols:
        qc_col = df_cpt.columns[df_cols.index('qc')]
        qc = df_cpt[qc_col].values
        col_map['Qc'] = qc_col
    else:
        print(f"Required column 'Qc' or 'qc' not found in {file_name}. Skipping.")
        return None

    if 'fs' in df_cols:
        fs_col = df_cpt.columns[df_cols.index('fs')]
        fs = df_cpt[fs_col].values
        col_map['Fs'] = fs_col
    else:
        print(f"Required column 'Fs' or 'fs' not found in {file_name}. Using Fs = 0.")
        fs = np.zeros(len(df_cpt))

    if 'u' in df_cols:
        u_col = df_cpt.columns[df_cols.index('u')]
        u = df_cpt[u_col].values
        col_map['U'] = u_col
    else:
        print(f"Required column 'U' or 'u' not found in {file_name}. Using U = 0.")
        u = np.zeros(len(df_cpt))

    depth = df_cpt[col_map['Depth']].values if 'Depth' in col_map else np.arange(len(df_cpt)) # Should not happen if filtered earlier

    # Filter out rows where Qc is not positive before calculation
    positive_qc_mask = (qc > 0)
    if positive_qc_mask.sum() < 2: # Need at least two points for analysis
        print(f"Warning: Less than two valid Qc points in {file_name} after filtering. Skipping layer analysis.")
        return None

    # Apply mask to all arrays
    df_cpt = df_cpt[positive_qc_mask].copy()
    qc = qc[positive_qc_mask]
    fs = fs[positive_qc_mask]
    u = u[positive_qc_mask]
    depth = depth[positive_qc_mask]
    
    # --- Geotechnical Calculations (Ic, Qtn, etc.) ---
    pa = 0.1 # MPa
    ground_water_level = 1.0 # m
    net_area_ratio = 0.8
    qt = qc + u * (1 - net_area_ratio)
    default_gamma = 0.00981 * 1.9 # kN/m^3 * 1.9 t/m^3 (approx density) -> MPa/m
    gamma_w = 9.80665 # kN/m^3
    
    # Calculate initial parameters
    Rf_safe = np.where(qt <= 0, np.nan, (fs / qt) * 100)
    gamma = ((0.27 * np.log10(np.where(Rf_safe > 0, Rf_safe, np.nan)) + 0.36 * np.log10(np.where(qt / pa > 0, qt / pa, np.nan)) + 1.236) * gamma_w) / 1000
    gamma = np.where((qc <= 0) | (fs <= 0) | np.isnan(gamma), default_gamma, gamma)
    gamma = np.maximum(14.0 / 1000, gamma) # minimum density 14kN/m^3

    totalStress = np.zeros(len(depth))
    u0 = np.zeros(len(depth))
    if len(depth) > 0:
        totalStress[0] = gamma[0] * depth[0]
        for i in range(1, len(depth)):
            totalStress[i] = totalStress[i - 1] + gamma[i] * (depth[i] - depth[i - 1])
            if depth[i] >= ground_water_level:
                u0[i] = 0.00981 * (depth[i] - ground_water_level)

    effStress = totalStress - u0
    Fr_safe = np.where(qt - totalStress <= 0, np.nan, (fs / (qt - totalStress)) * 100)
    Ic = np.zeros(len(depth))
    n = 0.5 * np.ones(len(depth))

    # Calculate Ic (Iterative process for exponent n)
    max_iterations = 1000
    for i in range(len(n)):
        deltan = 1
        iteration_counter = 0
        
        # Check for non-positive effective stress or invalid inputs early
        if effStress[i] <= 0 or (qt[i] - totalStress[i]) <= 0 or Fr_safe[i] <= 0 or np.isnan(Fr_safe[i]):
             Ic[i] = np.nan
             n[i] = np.nan
             continue

        while deltan >= 0.01 and iteration_counter < max_iterations:
            n0 = n[i]
            cN = (pa / effStress[i]) ** n[i] if effStress[i] > 0 else 1.7 # Use 1.7 if effStress is non-positive or 0
            
            if cN > 1.7:
                cN = 1.7
                
            Qtn_val = ((qt[i] - totalStress[i]) / pa) * cN
            
            # Check for non-positive Qtn_val which breaks log10
            if Qtn_val <= 0:
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
            print(f"Warning: Exceeded max iterations for file {file_name} at depth {depth[i]:.2f}m. Calculation may not have converged.")


    # --- Change-point Detection (Layer Analysis) ---
    df_analysis = pd.DataFrame({'Depth': depth, 'Qc': qc, 'Ic': Ic})
    df_analysis.dropna(subset=['Ic'], inplace=True)
    
    if df_analysis.empty or len(df_analysis) < 2:
        print(f"Warning: Insufficient valid Ic values for {file_name} after filtering. Skipping layer analysis.")
        return None

    data_for_segmentation = StandardScaler().fit_transform(df_analysis[['Qc', 'Ic']].values)
    depth_interval = np.median(np.diff(df_analysis['Depth'].values))
    
    if depth_interval == 0 or np.isnan(depth_interval):
        # Fallback if depth intervals are non-uniform or zero
        min_size = 5 # arbitrary minimum points
        print(f"Could not determine depth interval for {file_name}. Using min_size = {min_size}.")
    else:
        min_size = max(1, int(min_thickness_m / depth_interval))

    print(f"Using min_size of {min_size} data points (approx. {min_thickness_m} meters).")

    # The Pelt algorithm in ruptures is robust for unevenly spaced data, 
    # but the input must be time-ordered (which depth is).
    algo = rpt.Pelt(model="l2", jump=1, min_size=min_size).fit(data_for_segmentation)
    # Using a fixed penalty 'pen=5' might need calibration, but is a common starting point
    result = algo.predict(pen=5) 
    
    # 'result' contains the indices of the end of the segments (break points + last index)
    # Need to map the indices back to the original full dataframe (df_cpt)
    segment_indices = result
    
    # Map segmentation indices back to 'df_cpt'
    layer_indices_in_analysis_df = [0] + segment_indices
    if layer_indices_in_analysis_df[-1] != len(df_analysis):
         layer_indices_in_analysis_df[-1] = len(df_analysis)
         
    # Get the depths corresponding to the change points
    layer_depths = df_analysis.iloc[layer_indices_in_analysis_df[:-1]]['Depth'].values
    
    # Map these depths to the indices of the original full df_cpt (before analysis filtering)
    df_cpt_original_index = df_cpt[col_map['Depth']].index.tolist()
    
    layer_indices = []
    # Find the index in the original CPT file for each layer start depth
    for d in layer_depths:
        # Find the index in the original dataframe where Depth matches the change point depth
        matching_indices = df_cpt.index[df_cpt[col_map['Depth']] == d].tolist()
        if matching_indices:
            # We take the first match if multiple rows have the same depth
            layer_indices.append(matching_indices[0])

    # Ensure 0 (start) and the very end of the CPT data are included in the layer boundaries
    if df_cpt_original_index[0] not in layer_indices:
        layer_indices.insert(0, df_cpt_original_index[0])
    if df_cpt_original_index[-1] + 1 not in layer_indices:
        # Use the index *after* the last row for slicing
        layer_indices.append(df_cpt_original_index[-1] + 1)
    
    layer_indices = sorted(list(set(layer_indices)))

    # --- Plotting ---
    file_name_prefix = os.path.splitext(file_name)[0]
    fig, ax1 = plt.subplots(figsize=(8, 10))
    ax1.plot(df_analysis['Ic'].values, df_analysis['Depth'].values, label='Ic', color='blue')
    ax1.set_xlabel('Ic (blue) & Qc (red)')
    ax1.set_ylabel('Depth (m)')
    ax1.invert_yaxis()
    ax2 = ax1.twiny()
    ax2.plot(df_analysis['Qc'].values, df_analysis['Depth'].values, label='Qc', color='red')
    ax2.set_xlabel('Qc (MPa)')
    ax1.set_title(f"Qc & Ic with Layers Detected in {file_name}")
    
    # Draw horizontal lines for the layers (using the depths corresponding to the indices)
    layer_depths_for_plot = [df_cpt.loc[idx, col_map['Depth']] for idx in layer_indices[:-1] if idx < len(df_cpt)]
    for depth_val in layer_depths_for_plot:
        ax1.axhline(y=depth_val, color='r', linestyle='--', linewidth=1.5)
        
    plt.tight_layout()
    plot_path = os.path.join(output_plot_folder, f"{file_name_prefix}_layers_plot_qc_Ic.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Layer plot saved to {plot_path}")

    # --- Layer Statistics Calculation ---
    stats_data = []
    # Re-align df_cpt with calculated parameters (if needed for other stats, though mainly using original columns)
    df_cpt['Ic_calculated'] = np.nan
    df_cpt.loc[df_analysis.index, 'Ic_calculated'] = df_analysis['Ic'] 

    # Find all numeric columns in the original CPT data (plus the calculated 'Ic_calculated')
    numeric_cols = [col for col in df_cpt.select_dtypes(include=np.number).columns if col not in [col_map['Depth'], 'Layer']]
    
    for i in range(len(layer_indices) - 1):
        start_idx = layer_indices[i]
        end_idx = layer_indices[i + 1]
        layer_df = df_cpt.loc[start_idx:end_idx-1]
        
        if layer_df.empty:
            continue
            
        stats = {
            'Layer': i + 1,
            'Start Depth': layer_df[col_map['Depth']].iloc[0] if not layer_df.empty else np.nan,
            # End depth is the depth of the last point in the layer
            'End Depth': layer_df[col_map['Depth']].iloc[-1] if not layer_df.empty else np.nan,
        }
        stats['Thickness'] = stats['End Depth'] - stats['Start Depth'] + (depth_interval if depth_interval > 0 else 0)

        for col in numeric_cols:
             stats[f'{col} (Geometric Mean)'] = geometric_mean(layer_df[col])
             
        stats_data.append(stats)
        
    stats_df = pd.DataFrame(stats_data)
    stats_file_path = os.path.join(output_csv_folder, f"{file_name_prefix}_layers_stats.csv")
    stats_df.to_csv(stats_file_path, index=False)
    print(f"Layer statistics saved to {stats_file_path}")
    
    return stats_df

# -------------------------------------------------------------
# Mode 3: Merge All Measured Vs into CPT Profiles (Step Profile)
# -------------------------------------------------------------
def merge_all_profiles(Estimated_Vs_cpt_folder, Vs_folder, output_folder):
    indiv_folder = os.path.join(output_folder, "individual")
    os.makedirs(indiv_folder, exist_ok=True)
    final_df = pd.DataFrame()
    cpt_files = glob.glob(os.path.join(Estimated_Vs_cpt_folder, '*.csv'))
    vs_dict = {extract_key(os.path.basename(f)): f for f in glob.glob(os.path.join(Vs_folder, '*.csv'))}

    for qc_file in cpt_files:
        fname = os.path.basename(qc_file)
        code = extract_key(fname)
        
        try:
            df_qc = pd.read_csv(qc_file)
        except Exception as e:
            print(f"Error reading {qc_file}: {e}. Skipping file.")
            continue

        # Find case-insensitive qc and fs columns
        df_qc_cols_lower = [col.lower() for col in df_qc.columns]
        
        qc_cols = [col for col in df_qc.columns if col.lower() == "qc"]
        if qc_cols:
            df_qc["qc"] = df_qc[qc_cols].bfill(axis=1).iloc[:, 0]
        else:
             print(f"Warning (Mode 3): 'qc' column not found in {fname}. Skipping file.")
             continue

        fs_cols = [col for col in df_qc.columns if col.lower().startswith("fs")]
        if fs_cols:
            df_qc["fs"] = df_qc[fs_cols].bfill(axis=1).iloc[:, 0]
        # fs is not strictly required for filtering, but good to have if present

        # =============================================================
        # CPT Data Filtering (Depth > 0, Qc > 0, Fs > 0)
        # =============================================================
        initial_rows = len(df_qc)
        if 'Depth' in df_qc.columns:
            df_qc['Depth'] = df_qc['Depth'].round(3)
            df_qc = df_qc[df_qc['Depth'] > 0].copy()
            if len(df_qc) < initial_rows:
                print(f"Info (Mode 3): {initial_rows - len(df_qc)} rows with Depth = 0 were removed from {fname}.")
        initial_rows = len(df_qc) # Reset initial_rows count after Depth filter

        if "qc" in df_qc.columns:
            df_qc = df_qc[df_qc['qc'] > 0].copy()
            if len(df_qc) < initial_rows:
                 print(f"Warning (Mode 3): {initial_rows - len(df_qc)} rows with non-positive qc were removed from {fname}.")
        
        if "fs" in df_qc.columns:
            df_qc = df_qc[df_qc['fs'] > 0].copy()
            
        if df_qc.empty:
            print(f"Warning (Mode 3): After initial filtering, {fname} is empty. Skipping file.")
            continue
        # =============================================================

        if code in vs_dict:
            df_vs = pd.read_csv(vs_dict[code]).sort_values('d').reset_index(drop=True)
            
            # =============================================================
            # DEPTH LIMITATION LOGIC: Limit CPT data to max Vs depth (Mode 3)
            # =============================================================
            max_vs_depth = df_vs['d'].max()
            initial_qc_rows = len(df_qc)
            df_qc = df_qc[df_qc['Depth'] <= max_vs_depth ].copy()
            
            if len(df_qc) < initial_qc_rows:
                print(f"Info (Mode 3): {initial_qc_rows - len(df_qc)} CPT rows exceeding max Vs depth ({max_vs_depth:.2f}m) were removed from {fname}.")

            if df_qc.empty:
                print(f"Warning (Mode 3): After Vs depth filtering, {fname} is empty. Skipping Vs merge.")
                continue
            # =============================================================

            assign_vs = get_vs_step_assigner(df_vs)
            df_qc['Measure Vs'] = df_qc['Depth'].apply(assign_vs)
        else:
            df_qc['Measure Vs'] = None

        # Reorder columns: keep original columns, ensure 'qc', 'fs', 'Depth', 'Measure Vs' are prominent
        cols = list(df_qc.columns)
        primary_cols = ['File Name', 'Depth', 'qc', 'fs', 'Measure Vs']
        
        # Build the final column list, ensuring primary_cols are first (if present), followed by others
        new_cols = []
        for col in primary_cols:
            if col in cols:
                new_cols.append(col)
        
        for col in cols:
            if col not in new_cols:
                 new_cols.append(col)
        
        # Insert 'File Name' at the start for the individual file (it's inserted later for combined)
        df_qc.insert(0, "File Name", fname) 
        
        # Remove 'File Name' from new_cols for the subsequent subsetting to avoid issues
        final_new_cols = [c for c in new_cols if c != 'File Name']
        
        # Apply the final column order to the filtered DataFrame
        # Handle case where the original 'qc' or 'fs' might be kept along with the standardized ones
        final_df_qc = df_qc.filter(items=['File Name'] + final_new_cols)

        out_path = os.path.join(indiv_folder, f"processed_{code}_merged.csv")
        final_df_qc.to_csv(out_path, index=False)

        final_df = pd.concat([final_df, final_df_qc], ignore_index=True)

    output_path = os.path.join(output_folder, 'Mode3_combined_results.csv')
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"All files have been merged and saved to: {output_path}")

# -------------------------------------------------------------
# Mode 2: Layer Analysis + Step Profile Matching
# -------------------------------------------------------------
def run_mode2_workflow(Estimated_Vs_cpt_folder, Vs_folder, output_folder):
    print("\n[Vs-CPT Data Merging and Layer Analysis Mode]")
    print("\n--- Current Execution Information ---")
    print(f"Input CPT Folder: {os.path.abspath(Estimated_Vs_cpt_folder)}")
    print(f"Input Vs Folder: {os.path.abspath(Vs_folder)}")
    
    # Folder check
    if not os.path.isdir(Estimated_Vs_cpt_folder) or not os.path.isdir(Vs_folder):
        print("Error: The specified CPT or Vs folder does not exist.")
        print("Please check the paths. It is recommended to use quotation marks for the full path.")
        sys.exit(1)
        
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    main_output_dir = os.path.join(output_folder, f"Mode2_Analysis_Results_{timestamp}")
    
    analysis_csv_folder = os.path.join(main_output_dir, "Layer_Analysis_CSVs")
    analysis_plot_folder = os.path.join(main_output_dir, "Layer_Analysis_Plots")
    merged_results_folder = os.path.join(main_output_dir, "Merged_Results")
    
    os.makedirs(main_output_dir, exist_ok=True)
    os.makedirs(analysis_csv_folder, exist_ok=True)
    os.makedirs(analysis_plot_folder, exist_ok=True)
    os.makedirs(merged_results_folder, exist_ok=True)
    
    print("\n=============================================================")
    print("Starting Vs & CPT Integrated Analysis.")
    print("=============================================================")
    
    cpt_files = glob.glob(os.path.join(Estimated_Vs_cpt_folder, '*.csv'))
    vs_dict = {extract_key(os.path.basename(f)): f for f in glob.glob(os.path.join(Vs_folder, '*.csv'))}
    
    if not cpt_files:
        print("Warning: No CPT files found to analyze. Exiting program.")
        sys.exit(0)
        
    min_thickness_m = 0.5
    processed_cpt_dataframes = {}

    print("\n--- Step 1: Analyzing CPT files for layers and generating statistics ---")
    for cpt_file in cpt_files:
        file_name = os.path.basename(cpt_file)
        unique_code = extract_key(file_name)
        print(f"Processing file: {file_name}")
        
        if unique_code not in vs_dict:
            print(f"Warning: Vs data file for {file_name} does not exist. Skipping CPT file.")
            continue
            
        try:
            df_cpt = pd.read_csv(cpt_file)
            df_vs = pd.read_csv(vs_dict[unique_code])
        except Exception as e:
            print(f"Error reading {cpt_file} or Vs file: {e}. Skipping file.")
            continue
            
        if 'Depth' not in df_cpt.columns:
            print(f"Warning: 'Depth' column not found in CPT file {file_name}. Skipping.")
            continue
            
        # =============================================================
        # CRITICAL ADDITION: Filter CPT data to max Vs depth (Mode 2)
        # =============================================================
        max_vs_depth = df_vs['d'].max() if 'd' in df_vs.columns else 0
        
        if max_vs_depth <= 0:
            print(f"Warning: Vs file for {file_name} has invalid or non-positive max depth. Skipping CPT file.")
            continue
            
        initial_cpt_rows = len(df_cpt)
        df_cpt_filtered = df_cpt[df_cpt['Depth'] <= max_vs_depth].copy()
        
        if len(df_cpt_filtered) < initial_cpt_rows:
            print(f"Info (Mode 2): {initial_cpt_rows - len(df_cpt_filtered)} CPT rows exceeding max Vs depth ({max_vs_depth:.2f}m) were removed from {file_name}.")
            
        if df_cpt_filtered.empty:
            print(f"Warning (Mode 2): CPT data for {file_name} is empty after Vs depth filtering. Skipping layer analysis.")
            continue
        # =============================================================
        
        # Perform layer analysis on the filtered CPT data
        stats_df = analyze_cpt_for_layers_df(df_cpt_filtered, file_name, analysis_csv_folder, analysis_plot_folder, min_thickness_m)
        
        if stats_df is not None:
             processed_cpt_dataframes[unique_code] = (stats_df, df_vs)
             
        print("-" * 50)
        
    print("\n--- Step 1 Complete ---")
    print(f"Layer analysis result CSVs are in '{analysis_csv_folder}', and plots are in '{analysis_plot_folder}'.")
    print("\n--- Step 2: Merging analyzed CSV files with Vs data to create final results ---")
    
    final_df = pd.DataFrame()
    
    if not processed_cpt_dataframes:
        print("Warning: No CPT files were successfully analyzed. Final results were not generated.")
        return

    for unique_code, (df_cpt_processed, df_vs) in processed_cpt_dataframes.items():
        
        vs_file_name = os.path.basename(vs_dict[unique_code])
        df_cpt_processed.insert(0, "File Name", vs_file_name)
        
        if 'Start Depth' not in df_cpt_processed.columns or 'End Depth' not in df_cpt_processed.columns:
             print(f"Warning: Processed file for {unique_code} is missing 'Start Depth' or 'End Depth' columns. Skipping merge.")
             continue
             
        # Vs assignment
        assign_vs = get_vs_step_assigner(df_vs)
        
        # Add Depth column as mid-depth for each layer
        df_cpt_processed['Depth'] = (df_cpt_processed['Start Depth'] + df_cpt_processed['End Depth']) / 2
        df_cpt_processed['Measured Vs'] = df_cpt_processed['Depth'].apply(assign_vs)
        
        # Reorder columns
        cols = list(df_cpt_processed.columns)
        reordered_cols = ['File Name', 'Layer', 'Start Depth', 'End Depth', 'Depth', 'Measured Vs']
        for col in cols:
            if col not in reordered_cols:
                reordered_cols.append(col)
        df_cpt_processed = df_cpt_processed[reordered_cols]
        
        individual_merged_path = os.path.join(merged_results_folder, f"Processed_{unique_code}_merged_layers.csv")
        df_cpt_processed.to_csv(individual_merged_path, index=False, encoding='utf-8-sig')
        print(f"Individual merged result saved: '{os.path.basename(individual_merged_path)}'")
        
        final_df = pd.concat([final_df, df_cpt_processed], ignore_index=True)
            
    if not final_df.empty:
        output_path = os.path.join(main_output_dir, 'Mode2_combined_results.csv')
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Final combined results have been successfully saved to '{output_path}'.")
    else:
        print("Warning: No files to merge. Final results were not generated.")
        
    print("\n--- Step 2 Complete ---")
    print("\nAll analysis and merging are complete. Result files are organized in the output folder.")

# -------------------------------------------------------------
# Main Execution Block
# -------------------------------------------------------------
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
