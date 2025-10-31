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
  
  *FIXED: Measured Vs expansion logic is moved outside the Vs model loop
          to ensure consistent plotting across all Vs models for one CPT.*

Usage:
  Run the script from the command line with the three folder paths
  as arguments. Use 'N' to skip a folder if the data source is
  not available.

  python your_script_name.py <Processed_Vs_Folder> <Measured_Vs_Folder> <CPT_Vs_Folder>
  
  Example:
  python comparison.py ./results_Qc_Ic_1 ./vsmo ./cpt_data
=============================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from datetime import datetime

# -----------------------------------------------------------
# Check command-line arguments
# -----------------------------------------------------------
if len(sys.argv) != 4:
    print("Usage: python your_script_name.py <Processed_Vs_Folder> <Measured_Vs_Folder> <CPT_Vs_Folder>")
    print("Example: python comparison.py ./results_Qc_Ic_1 ./vsmo ./cpt_data")
    print("Use 'N' to skip a folder.")
    sys.exit(1)

processed_vs_folder = sys.argv[1] if sys.argv[1].upper() != 'N' else None
measured_vs_folder = sys.argv[2] if sys.argv[2].upper() != 'N' else None
cpt_vs_folder = sys.argv[3] if sys.argv[3].upper() != 'N' else None

# -----------------------------------------------------------
# Output folder setup
# -----------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f'./comparison_plots_output_{timestamp}'
os.makedirs(output_folder, exist_ok=True)

# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------
def extract_file_key(filename):
    """Extract consistent key from filename"""
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) >= 3:
        return parts[1]
    elif len(parts) == 2:
        return parts[1].replace('.csv', '')
    return None

def get_file_map(folder):
    """Create dictionary mapping code -> file path"""
    file_map = {}
    if folder and os.path.exists(folder):
        for f in os.listdir(folder):
            if f.endswith('.csv'):
                code = extract_file_key(f)
                if code:
                    file_map[code] = os.path.join(folder, f)
    return file_map

# -----------------------------------------------------------
# Build file maps
# -----------------------------------------------------------
processed_file_map = get_file_map(processed_vs_folder)
measured_file_map = get_file_map(measured_vs_folder)
cpt_file_map = get_file_map(cpt_vs_folder)
processed_codes = sorted(list(processed_file_map.keys()))

# -----------------------------------------------------------
# Columns to compare
# -----------------------------------------------------------
cpt_vs_cols = [
    'andrus_2007_holocene Vs',
    'andrus_2007_pleistocene Vs',
    'andrus_2007_tertiary_age_cooper_marl Vs',
    'robertson_2009 Vs',
    'hegazy_2006 Vs',
    'mcgann_2015 Vs',
    'mcgann_2018 Vs'
]
processed_vs_cols = [
    'andrus_2007_holocene Vs (Geometric Mean)',
    'andrus_2007_pleistocene Vs (Geometric Mean)',
    'andrus_2007_tertiary_age_cooper_marl Vs (Geometric Mean)',
    'robertson_2009 Vs (Geometric Mean)',
    'hegazy_2006 Vs (Geometric Mean)',
    'mcgann_2015 Vs (Geometric Mean)',
    'mcgann_2018 Vs (Geometric Mean)'
]

# -----------------------------------------------------------
# Main Loop
# -----------------------------------------------------------
for code in processed_codes:
    print(f"Processing code: {code}")
    
    df_processed = None
    df_measured = None
    df_cpt = None

    try:
        if code in processed_file_map:
            df_processed = pd.read_csv(processed_file_map[code])
        if code in measured_file_map:
            df_measured = pd.read_csv(measured_file_map[code])
        if code in cpt_file_map:
            df_cpt = pd.read_csv(cpt_file_map[code])
    except Exception as e:
        print(f"Skipping code '{code}' due to file error: {e}")
        continue

    # =======================================================
    # ✅ FIX: Measured Vs 데이터 확장 로직을 Vs 모델 루프 밖으로 이동
    #        (한 CPT 파일에 대해 한 번만 확장되도록 보장)
    # =======================================================
    if df_measured is not None and 'd' in df_measured.columns and 'vs' in df_measured.columns:
        df_measured = df_measured.sort_values('d').reset_index(drop=True).copy() # 명시적 복사본 생성
        
        # 첫 번째 깊이가 0보다 크면 median Δd 기반으로 확장
        if not df_measured.empty and df_measured.iloc[0]['d'] > 0:
            depth_diffs = np.diff(df_measured['d'].values)
            # 깊이 차이가 없으면 (단일 데이터 포인트) 첫 깊이를 미디안으로 사용
            median_diff = np.median(depth_diffs) if len(depth_diffs) > 0 else df_measured.iloc[0]['d']
            
            # 새로운 시작 깊이 계산 (0보다 작아지지 않도록)
            new_depth = max(df_measured.iloc[0]['d'] - median_diff, 0.0)
            
            # 새로운 행을 데이터프레임 맨 앞에 추가 (한 번만 실행)
            new_row = pd.DataFrame([{'d': new_depth, 'vs': df_measured.iloc[0]['vs']}])
            df_measured = (
                pd.concat([new_row, df_measured], ignore_index=True)
                .sort_values('d')
                .reset_index(drop=True)
            )

    # -------------------------------------------------------
    # Determine max Vs for scaling
    # -------------------------------------------------------
    max_vs_for_code = 0
    if df_cpt is not None:
        for col in cpt_vs_cols:
            if col in df_cpt.columns:
                max_vs_for_code = max(max_vs_for_code, df_cpt[col].max())

    if df_processed is not None:
        for col in processed_vs_cols:
            if col in df_processed.columns:
                max_vs_for_code = max(max_vs_for_code, df_processed[col].max())

    # *확장된 df_measured를 사용하여 max Vs 계산*
    if df_measured is not None and 'vs' in df_measured.columns:
        max_vs_for_code = max(max_vs_for_code, df_measured['vs'].max())

    max_vs_for_code = max_vs_for_code * 1.1 if max_vs_for_code > 0 else 500

    # -------------------------------------------------------
    # Output subfolder
    # -------------------------------------------------------
    code_output_folder = os.path.join(output_folder, code)
    os.makedirs(code_output_folder, exist_ok=True)
    
    # -------------------------------------------------------
    # Plot each correlation model (내부 루프)
    # -------------------------------------------------------
    for col_cpt, col_processed in zip(cpt_vs_cols, processed_vs_cols):
        plt.figure(figsize=(6, 8))
        plt.title(f'Vs Profile Comparison for {code} - {col_cpt.replace(" Vs", "")}') # 제목 정리
        plt.xlabel('Vs (m/s)')
        plt.ylabel('Depth (m)')
        plt.gca().invert_yaxis()

        # --- CPT-estimated Vs (Blue line) ---
        if df_cpt is not None and col_cpt in df_cpt.columns and 'Depth' in df_cpt.columns:
            plt.plot(df_cpt[col_cpt], df_cpt['Depth'], label='CPT-estimated Vs (Point)', color='blue', linewidth=2)

        # --- Processed CPT Vs (Red step) ---
        if df_processed is not None and col_processed in df_processed.columns:
            end_depths = df_processed['End Depth'].values
            start_depths = np.insert(end_depths[:-1], 0, 0)
            vs_vals = df_processed[col_processed].values

            y_coords_processed = np.empty(2 * len(start_depths))
            y_coords_processed[0::2] = start_depths
            y_coords_processed[1::2] = end_depths
            x_coords_processed = np.repeat(vs_vals, 2)
            
            # 마지막 데이터 포인트의 끝 깊이를 명시적으로 설정
            if not np.isclose(y_coords_processed[-1], df_processed['End Depth'].iloc[-1]):
                 y_coords_processed[-1] = df_processed['End Depth'].iloc[-1]
            
            plt.step(x_coords_processed, y_coords_processed, where='post',
                     label='Processed CPT-estimated Vs (Layer)', color='red', linewidth=2)

        # --- Measured Vs (Green step, using the already expanded df_measured) ---
        if df_measured is not None and 'd' in df_measured.columns and 'vs' in df_measured.columns:
            
            vs2_depths = df_measured['d'].values
            vs2_vs_vals = df_measured['vs'].values
            
            # Start depths: [d_new, d1, d2, ...]
            vs2_start_depths = np.insert(vs2_depths[:-1], 0, vs2_depths[0]) 
            vs2_end_depths = vs2_depths

            # Measured Vs는 일반적으로 이미 층 경계(d)를 나타내므로, Step Plot을 구성
            y_coords_vs2 = np.empty(2 * len(vs2_start_depths))
            y_coords_vs2[0::2] = vs2_start_depths
            y_coords_vs2[1::2] = vs2_end_depths
            x_coords_vs2 = np.repeat(vs2_vs_vals, 2)

            plt.step(x_coords_vs2, y_coords_vs2, where='post',
                     label='Measured Vs (Field)', color='green', linestyle='--', linewidth=2)

        # --- Final plot formatting ---
        plt.xlim(0, max_vs_for_code)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        plot_filename = f'{code}_{col_cpt.replace(" ", "_")}.png'
        plt.savefig(os.path.join(code_output_folder, plot_filename))
        plt.close()

    print(f"Generated plots for code '{code}' and saved them to {code_output_folder}")

print("Script finished. All plots have been generated and saved.")
