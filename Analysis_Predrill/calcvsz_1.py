import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the VsProfile class from the module 'vs_calc'.
# NOTE: This script assumes you have a module named 'vs_calc' 
# with the 'VsProfile' class and the 'utils' submodule.
from vs_calc import VsProfile
from vs_calc import utils 

# 💡 중요: Vs 데이터 파일의 실제 깊이 간격을 변수로 명확히 정의 (1.0m 간격 가정)
VS_DATA_INTERVAL = 1.0 

# =========================================================================
# A. VsZ Smearing Calculation Function (1.0m 간격)
# =========================================================================

def calculate_vsz_with_smearing(target_depth=15, max_smear_m=15, source_dir="Vs_1m", output_filename="VsZ_15m_Smearing_Results_1m.csv"):
    """
    Reads Vs profiles, calculates VsZ (Z=15m) with smearing applied to 
    the upper layers (up to i+1m), and saves results to a single CSV file.

    *** Smearing depth 'i' is iterated in 1.0m intervals. ***
    """
    
    results = []
    vsz_calc_z = target_depth       
    
    # Smearing interval은 1.0m로 유지
    SMEAR_INTERVAL = 1.0 

    # --- Directory Check ---
    if not os.path.exists(source_dir):
         os.makedirs(source_dir)
         print(f"📁 Folder '{source_dir}' created. Please place your Vs files inside.")
         return
    # -----------------------

    csv_files = glob.glob(os.path.join(source_dir, "*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in the '{source_dir}' folder.")
        return

    print(f"🔍 Processing {len(csv_files)} CSV files. (Target VsZ Depth: {vsz_calc_z}m)")

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        profile_name = os.path.splitext(file_name)[0]
        
        row_data = {"Profile_Name": profile_name, "File_Name": file_name}
        
        try:
            df = pd.read_csv(file_path)
            
            # 1. Data Preparation and Cropping to max_z_crop
            # Vs 데이터의 컬럼 이름 유연하게 처리
            depth_original = np.asarray(df['d']) if 'd' in df.columns else np.asarray(df['Depth'])
            vs_original = np.asarray(df['vs']) if 'vs' in df.columns else np.asarray(df['Vs'])
            vs_sd_original = np.asarray(df['Vs_SD']) if 'Vs_SD' in df.columns else np.zeros_like(vs_original)
            
            # Filter data only up to max_z_crop
            crop_mask = depth_original <= vsz_calc_z
            
            depth_cropped = depth_original[crop_mask]
            vs_cropped = vs_original[crop_mask]
            vs_sd_cropped = vs_sd_original[crop_mask]
            
            max_depth_after_crop = np.max(depth_cropped) if len(depth_cropped) > 0 else 0
            
            # 2. Check for sufficient depth
            if max_depth_after_crop < vsz_calc_z:
                 print(f"⚠️ {file_name}: Cropped max depth ({max_depth_after_crop}m) is shallower than target ({vsz_calc_z}m). Skipping calculation.")
                 row_data[f"VsZ_{vsz_calc_z}m_D0"] = np.nan
                 results.append(row_data)
                 continue

            num_layers = len(depth_cropped)

            # 3. Smearing Loop (i = 0, 1.0, 2.0, 3.0, ...)
            smear_depths = np.arange(0, max_smear_m + SMEAR_INTERVAL, SMEAR_INTERVAL)

            for i in smear_depths:
                
                i_rounded = round(i, 1)

                vs_modified = np.copy(vs_cropped)
                vs_sd_modified = np.copy(vs_sd_cropped)
                
                if i_rounded == 0.0:
                    column_name = f"VsZ_{vsz_calc_z}m_D0" 
                else:
                    # Representative depth (Z_rep = i + 1) in meters
                    Z_rep_m = i_rounded + 1.0
                    
                    # 🚀 수정: Vs 데이터 간격 (1.0m)을 사용하여 정확한 인덱스 계산
                    idx_rep = int(round(Z_rep_m / VS_DATA_INTERVAL))
                    idx_smear_end = idx_rep 
                    
                    if idx_rep >= num_layers - 1:
                        # 데이터 부족으로 인해 스미어링 중단
                        print(f"⚠️ {file_name}: Smearing stopped at i={i_rounded}m. Required Z_rep={Z_rep_m}m, but max data index is {num_layers-1}. (Data might be too shallow)")
                        break

                    V_s_rep = vs_modified[idx_rep]
                    
                    # Smear the upper layers
                    vs_modified[0 : idx_smear_end] = V_s_rep
                    vs_sd_modified[0 : idx_smear_end] = vs_sd_modified[idx_rep]
                    
                    # Column name D1, D2, D3, ...
                    column_name = f"VsZ_{vsz_calc_z}m_D{int(i_rounded)}"


                # Create VsProfile instance and calculate VsZ
                temp_profile = VsProfile(
                    name=profile_name,
                    vs=vs_modified,  
                    vs_sd=vs_sd_modified, 
                    depth=depth_cropped, 
                )

                temp_profile.max_depth = vsz_calc_z
                vsz_value = temp_profile.calc_vsz()
                
                row_data[column_name] = round(vsz_value, 3)
                
            # 4. Add the final result row to the list
            results.append(row_data)

        except KeyError as e:
            print(f"❌ {file_name}: Missing required column ({e}). Requires 'd'/'Depth', 'vs'/'Vs', and 'Vs_SD' columns.")
        except Exception as e:
            print(f"❌ {file_name}: Unexpected error during processing: {e}")

    # 5. Save combined results to CSV
    if results:
        results_df = pd.DataFrame(results)
        # Sort columns to ensure D0 is first, D1 is second, D2 is third, etc.
        data_columns = [col for col in results_df.columns if f"VsZ_{vsz_calc_z}m_D" in col]
        
        # 정렬을 위해 D 뒤의 숫자를 float으로 변환하여 사용
        def sort_key(col_name):
            part = col_name.split('_D')[1].replace('_', '.')
            if part == '0':
                return 0.0
            return float(part)

        sorted_columns = sorted(data_columns, key=sort_key)
        
        other_columns = [col for col in results_df.columns if col not in data_columns]
        results_df = results_df[other_columns + sorted_columns]

        output_path = os.path.join(source_dir, output_filename)
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n🎉 All calculation results saved to '{output_path}'.")
        return output_path 
    else:
        print("\nNo results to save.")
        return None

# -------------------------------------------------------------------------
# B. Residual Analysis and Plotting Function (1.0m 간격)
# -------------------------------------------------------------------------

def analyze_and_plot_vsz_smearing_residuals(file_path, target_depth=15, max_smear_m=15, source_dir="Vs_1m"):
    """
    Reads the Smearing calculation results, calculates the natural logarithm residual, and visualizes it.
    Residual = ln(VsZ_D0) - ln(VsZ_Di)
    """
    
    if not file_path or not os.path.exists(file_path):
        print(f"❌ Result file '{file_path}' not found. Skipping analysis.")
        return

    print("\n\n📊 Starting Residual Analysis and Visualization...")
    df = pd.read_csv(file_path)
    
    # 1. Calculate Residuals
    vsz_d0_col = f"VsZ_{target_depth}m_D0"
    
    if vsz_d0_col not in df.columns:
        print(f"❌ Baseline column '{vsz_d0_col}' not found. Cannot calculate residuals.")
        return

    log_vsz_d0 = np.log(df[vsz_d0_col])
    
    residual_data = []

    # Iterate over columns to dynamically extract smearing depths
    data_columns = [col for col in df.columns if col.startswith(f"VsZ_{target_depth}m_D") and col != vsz_d0_col]
    
    for vsz_di_col in data_columns:
        
        smear_depth_str = vsz_di_col.split('_D')[1].replace('_', '.')
        try:
            i = float(smear_depth_str)
            if i != int(i): 
                 continue
            i = int(i)
        except ValueError:
            print(f"⚠️ Could not parse smearing depth from column: {vsz_di_col}")
            continue
            
        log_vsz_di = np.log(df[vsz_di_col])
        residual = log_vsz_d0 - log_vsz_di
        
        temp_df = pd.DataFrame({
            "Residual": residual,
            "Smear_Depth_m": i, 
            "VsZ_Original": df[vsz_d0_col]
        }).dropna()
        residual_data.append(temp_df)
        
    if not residual_data:
        print("❌ No valid residual data available. Skipping visualization.")
        return
        
    residuals_df = pd.concat(residual_data)
    
    # 2. Visualize Residual Distribution
    
    # 2.1. Line Plot showing Median and Standard Deviation (Using Error Bars)
    
    residual_stats = residuals_df.groupby('Smear_Depth_m')['Residual'].agg(['median', 'std']).reset_index()
    std_dev = residual_stats['std'].values
    
    plt.figure(figsize=(12, 8)) 
    
    plt.errorbar(
        residual_stats['Smear_Depth_m'], 
        residual_stats['median'], 
        yerr=std_dev, 
        fmt='o',             
        color='blue', 
        ecolor='blue',           
        capsize=5, 
        markersize=10,       
        linewidth=2,         
        linestyle='None',    
        label='Median $\pm 1$ $\sigma$'
    )
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    # --- Symmetric Y-axis Calculation ---
    upper_bound = residual_stats['median'] + residual_stats['std']
    lower_bound = residual_stats['median'] - residual_stats['std']
    
    max_abs_y = np.ceil(np.max(np.abs(
        np.concatenate([
            upper_bound, 
            lower_bound,
            [0.01]
        ])
    )) * 10) / 10
    
    plt.ylim(-max_abs_y, max_abs_y) 
    
    # ---------------------------------------------
    
    plt.title(f'VsZ_{target_depth}m Residual: Median and Standard Deviation by Smearing Depth (1.0m intervals)', fontsize=16)
    plt.xlabel('Smearing Depth $i$ (m)', fontsize=14)
    plt.ylabel(r'Residual: $\ln(\text{VsZ}_0) - \ln(\text{VsZ}_i)$', fontsize=14)
    
    x_ticks = residual_stats['Smear_Depth_m'].values
    x_labels = [f'{int(t)}' for t in x_ticks]
    
    plt.xticks(x_ticks, labels=x_labels, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0, x_ticks.max() * 1.05) 
    
    plt.legend(loc='best', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save and display plot
    errorbar_plot_path = os.path.join(source_dir, "Residual_Median_ErrorBar_Plot_1m_Final.png")
    plt.savefig(errorbar_plot_path)
    plt.show()
    print(f"🖼️ Median/Error Bar Plot saved to '{errorbar_plot_path}'.")
    
    # --- (Scatter Plot) ---
    
    # 2.2. Scatter Plot 
    plt.figure(figsize=(12, 8)) 
    sns.scatterplot(
        x="VsZ_Original", 
        y="Residual", 
        hue="Smear_Depth_m", 
        data=residuals_df, 
        palette="viridis", 
        alpha=0.6,
        s=50 
    )
    plt.axhline(0, color='r', linestyle='--', linewidth=1, label='Zero Residual')
    
    plt.title(f'VsZ_{target_depth}m Residual vs. Original VsZ (1.0m Smearing intervals)', fontsize=16)
    plt.xlabel(r'Original $\text{VsZ}_0$ (m/s)', fontsize=14)
    plt.ylabel(r'Residual: $\ln(\text{VsZ}_0) - \ln(\text{VsZ}_i)$', fontsize=14)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.legend(title='Smear Depth $i$ (m)', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save and display plot
    scatter_plot_path = os.path.join(source_dir, "Residual_ScatterPlot_1m_Final.png")
    plt.savefig(scatter_plot_path)
    plt.show()
    print(f"🖼️ Scatter Plot saved to '{scatter_plot_path}'.")
    
    print("\n✅ Analysis and Visualization Complete.")

# =========================================================================
# C. Main Execution Block
# =========================================================================

if __name__ == "__main__":
    # max_smear_m=15으로 설정하여 D15m까지 계산 수행
    output_csv_path = calculate_vsz_with_smearing(target_depth=15, max_smear_m=15, source_dir="Vs_1m")
    
    if output_csv_path:
        analyze_and_plot_vsz_smearing_residuals(output_csv_path, target_depth=15, max_smear_m=15, source_dir="Vs_1m")