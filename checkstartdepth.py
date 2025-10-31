# -*- coding: utf-8 -*-
"""
=============================================================
Measured Vs Expansion Start Depth Calculation Script
-------------------------------------------------------------
Purpose:
  This script calculates the expanded starting depth (d_new)
  for Measured Vs profiles, using the median of the depth
  differences (Δd), ensuring d_new >= 0.
  The results are saved to a single CSV file.

Usage:
  Run the script from the command line with the folder path
  containing the Measured Vs CSV files as an argument.

  python vs_expansion_analysis.py <Measured_Vs_Folder>

Example:
  python vs_expansion_analysis.py ./vsmo
=============================================================
"""

import os
import pandas as pd
import numpy as np
import sys
from datetime import datetime

# -----------------------------------------------------------
# 1. Check command-line arguments and Setup
# -----------------------------------------------------------
if len(sys.argv) != 2:
    print("Usage: python vs_expansion_analysis.py <Measured_Vs_Folder>")
    print("Example: python vs_expansion_analysis.py ./vsmo")
    sys.exit(1)

measured_vs_folder = sys.argv[1]

if not os.path.exists(measured_vs_folder):
    print(f"Error: Folder not found at path: {measured_vs_folder}")
    sys.exit(1)

# Output folder setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f'./vs_expansion_output_{timestamp}'
os.makedirs(output_folder, exist_ok=True)
output_csv_filename = 'measured_vs_expansion_start_depths.csv'
output_csv_path = os.path.join(output_folder, output_csv_filename)

print(f"Processing Measured Vs files in: {measured_vs_folder}")
print(f"Results will be saved to: {output_csv_path}")

# -----------------------------------------------------------
# 2. Helper function to extract file key (code)
# -----------------------------------------------------------
def extract_file_key(filename):
    """Extract consistent key (code) from filename"""
    base = os.path.basename(filename)
    parts = base.split('_')
    # 파일명이 'vsmo_CODE.csv' 형태라고 가정하고 CODE를 추출
    if len(parts) >= 2:
        # .csv 확장자 제거
        return parts[1].replace('.csv', '')
    return None

# -----------------------------------------------------------
# 3. Main Calculation Loop
# -----------------------------------------------------------
vs_expansion_data = []

# 폴더 내의 모든 .csv 파일을 순회
for filename in os.listdir(measured_vs_folder):
    if not filename.endswith('.csv'):
        continue

    file_path = os.path.join(measured_vs_folder, filename)
    code = extract_file_key(filename)

    if not code:
        print(f"Warning: Could not extract code from file: {filename}. Skipping.")
        continue
    
    # Measured Vs 파일 로드
    try:
        df_measured = pd.read_csv(file_path)
    except Exception as e:
        print(f"Skipping file '{filename}' due to load error: {e}")
        continue
    
    # 필수 컬럼 ('d': 깊이, 'vs': Vs) 확인 및 처리
    if 'd' not in df_measured.columns or 'vs' not in df_measured.columns:
        print(f"Warning: File '{filename}' is missing 'd' or 'vs' column. Skipping.")
        continue

    df_measured = df_measured.sort_values('d').reset_index(drop=True)
    
    # 첫 번째 깊이가 0보다 클 경우에만 확장 시작 깊이 계산
    if not df_measured.empty and df_measured.iloc[0]['d'] > 0:
        first_depth = df_measured.iloc[0]['d']
        
        # 깊이 차이 계산: Δd = d[i] - d[i-1]
        depth_diffs = np.diff(df_measured['d'].values)
        
        # 깊이 차이가 없으면 (단일 데이터 포인트) 첫 깊이를 미디안으로 사용
        median_diff = np.median(depth_diffs) if len(depth_diffs) > 0 else first_depth
        
        # 새로운 시작 깊이 계산 (최솟값은 0)
        # new_depth = max(df_measured.iloc[0]['d'] - median_diff, 0.0)
        new_depth = max(first_depth - median_diff, 0.0)
        
        # 결과를 리스트에 추가
        vs_expansion_data.append({
            'File Code': code,
            'Original Filename': filename,
            'First Measured Depth (m)': first_depth,
            'Depth Difference Median (m)': median_diff,
            'Expanded Start Depth (d_new, m)': new_depth
        })
        print(f"Processed {code}: d_new = {new_depth:.3f} m")
    else:
        # 첫 깊이가 이미 0이거나 데이터가 비어있는 경우
        vs_expansion_data.append({
            'File Code': code,
            'Original Filename': filename,
            'First Measured Depth (m)': df_measured.iloc[0]['d'] if not df_measured.empty else np.nan,
            'Depth Difference Median (m)': 0.0,
            'Expanded Start Depth (d_new, m)': 0.0
        })
        # print(f"Skipping expansion for {code}: First depth is 0 or data is empty.")

# -----------------------------------------------------------
# 4. Save Results
# -----------------------------------------------------------
if vs_expansion_data:
    df_vs_expansion = pd.DataFrame(vs_expansion_data)
    
    # CSV 파일로 저장
    df_vs_expansion.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 분석이 완료되었습니다. 결과가 저장되었습니다: {output_csv_path}")
else:
    print("\n⚠️ 처리할 Measured Vs CSV 파일이 Measured Vs 폴더 내에 없습니다.")