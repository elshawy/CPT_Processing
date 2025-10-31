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

# =========================================================================
# A. VsZ Smearing Calculation Function (Unchanged)
# =========================================================================

def calculate_vsz_with_smearing(target_depth=15, max_smear_m=5, source_dir="Vs", output_filename="VsZ_15m_Smearing_Results.csv"):
    """
    Reads Vs profiles, calculates VsZ (Z=15m) with smearing applied to 
    the upper layers (up to i+1m), and saves results to a single CSV file.

    1. Crop data in each CSV file up to the target_depth.
    2. Calculate VsZ by applying Smearing: replacing the Vs values of the 
       upper layers (up to i+1m) with the Vs value of the layer below.
    3. Save the results to a combined CSV file.

    Args:
        target_depth (int): The target depth Z for VsZ calculation (e.g., 15m).
        max_smear_m (int): The maximum index 'i' for smearing (e.g., i=5 means Smearing up to 5m).
        source_dir (str): Path to the folder containing CSV files.
        output_filename (str): Name of the output CSV file for results.
    """
    
    results = []
    max_z_crop = target_depth      
    vsz_calc_z = target_depth       
    max_smear_index = max_smear_m  

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
            depth_original = np.asarray(df['d']) if 'd' in df.columns else np.asarray(df['Depth'])
            vs_original = np.asarray(df['vs']) if 'vs' in df.columns else np.asarray(df['Vs'])
            vs_sd_original = np.asarray(df['Vs_SD']) if 'Vs_SD' in df.columns else np.zeros_like(vs_original)
            
            # Filter data only up to max_z_crop
            crop_mask = depth_original <= max_z_crop
            
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

            # 3. Smearing Loop (i = 0, 1, 2, 3...)
            for i in range(max_smear_index + 1):
                
                vs_modified = np.copy(vs_cropped)
                vs_sd_modified = np.copy(vs_sd_cropped)
                
                if i == 0:
                    smear_info = "Original (No Smearing)"
                else:
                    # Representative depth (Z_rep = i + 1) and index (idx_rep = Z_rep)
                    Z_rep = i + 1 
                    idx_rep = Z_rep 
                    idx_smear_end = Z_rep 
                    
                    if idx_rep >= num_layers - 1:
                        break

                    V_s_rep = vs_modified[idx_rep]
                    
                    # Smear the upper layers: Replace Vs values of layers from 0m up to (Z_rep-1)m
                    vs_modified[0 : idx_smear_end] = V_s_rep
                    vs_sd_modified[0 : idx_smear_end] = vs_sd_modified[idx_rep]
                    
                    smear_info = f"{i}m Smear (Vs in 0~{Z_rep}m range replaced by {V_s_rep:.3f} m/s)"

                # Create VsProfile instance and calculate VsZ
                temp_profile = VsProfile(
                    name=profile_name,
                    vs=vs_modified,  
                    vs_sd=vs_sd_modified, 
                    depth=depth_cropped, 
                )

                temp_profile.max_depth = vsz_calc_z
                vsz_value = temp_profile.calc_vsz()
                
                column_name = f"VsZ_{vsz_calc_z}m_D{i}"
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
        output_path = os.path.join(source_dir, output_filename)
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n🎉 All calculation results saved to '{output_path}'.")
        return output_path 
    else:
        print("\nNo results to save.")
        return None

# -------------------------------------------------------------------------
# B. Residual Analysis and Plotting Function (Median and Std. Dev. Plot)
# -------------------------------------------------------------------------

def analyze_and_plot_vsz_smearing_residuals(file_path, target_depth=15, max_smear_m=5, source_dir="Vs"):
    """
    Reads the Smearing calculation results, calculates the natural logarithm residual, and visualizes it.
    Residual = ln(VsZ_D0) - ln(VsZ_Di)
    
    Args:
        file_path (str): Path to the CSV file containing VsZ results.
        target_depth (int): The target depth Z for VsZ (e.g., 15m).
        max_smear_m (int): Maximum smearing depth (i).
        source_dir (str): Directory for saving plot images.
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

    for i in range(1, max_smear_m + 1):
        vsz_di_col = f"VsZ_{target_depth}m_D{i}"
        
        if vsz_di_col not in df.columns:
            print(f"⚠️ Warning: Column {vsz_di_col} is missing. Stopping residual calculation.")
            break
            
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
    
    # Calculate median and standard deviation per Smearing Depth
    residual_stats = residuals_df.groupby('Smear_Depth_m')['Residual'].agg(['median', 'std']).reset_index()
    std_dev = residual_stats['std'].values
    
    plt.figure(figsize=(10, 6))
    
    # Plot Median Residual with error bars for +/- 1 Standard Deviation
    plt.errorbar(
        residual_stats['Smear_Depth_m'], 
        residual_stats['median'], 
        yerr=std_dev, 
        fmt='o',             # Format: circle markers
        color='blue', 
        ecolor='blue',           # Error bar color (set to blue)
        capsize=5, 
        markersize=8,        # Increased marker size
        linewidth=2,         # Increased line width for tails
        linestyle='None',    # REMOVED the line connecting median points
        label='Median $\pm 1$ $\sigma$'
    )
    
    # Axes setup and labels
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    # --- FIX for Symmetric Y-axis Calculation ---
    
    # Calculate the upper and lower bounds of the 1-sigma range (using existing columns)
    upper_bound = residual_stats['median'] + residual_stats['std']
    lower_bound = residual_stats['median'] - residual_stats['std']
    
    # Calculate the maximum absolute residual value for symmetric y-axis
    max_abs_y = np.ceil(np.max(np.abs(
        np.concatenate([
            upper_bound, 
            lower_bound,
            [0.01] # Ensures a minimum range 
        ])
    )) * 10) / 10 # Rounds up to the nearest 0.1 for clean scaling
    
    # Apply Symmetric Y-axis range
    plt.ylim(-max_abs_y, max_abs_y) 
    
    # ---------------------------------------------
    
    plt.title(f'VsZ_{target_depth}m Residual: Median and Standard Deviation by Smearing Depth')
    plt.xlabel('Smearing Depth $i$ (m)')
    plt.ylabel(r'Residual: $\ln(\text{VsZ}_0) - \ln(\text{VsZ}_i)$')
    plt.xticks(residual_stats['Smear_Depth_m']) 
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save and display plot
    errorbar_plot_path = os.path.join(source_dir, "Residual_Median_ErrorBar_Plot_Final.png")
    plt.savefig(errorbar_plot_path)
    plt.show()
    print(f"🖼️ Median/Error Bar Plot saved to '{errorbar_plot_path}'.")
    
    # ---
    
    # 2.2. Scatter Plot (Retained)
    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        x="VsZ_Original", 
        y="Residual", 
        hue="Smear_Depth_m", 
        data=residuals_df, 
        palette="viridis", 
        alpha=0.6
    )
    plt.axhline(0, color='r', linestyle='--', linewidth=1, label='Zero Residual')
    
    plt.title(f'VsZ_{target_depth}m Residual vs. Original VsZ')
    plt.xlabel(r'Original $\text{VsZ}_0$ (m/s)')
    plt.ylabel(r'Residual: $\ln(\text{VsZ}_0) - \ln(\text{VsZ}_i)$')
    
    # Detailed legend for the Smearing Depth
    plt.legend(title='Smear Depth $i$ (m)', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save and display plot
    scatter_plot_path = os.path.join(source_dir, "Residual_ScatterPlot.png")
    plt.savefig(scatter_plot_path)
    plt.show()
    print(f"🖼️ Scatter Plot saved to '{scatter_plot_path}'.")
    
    print("\n✅ Analysis and Visualization Complete.")

# =========================================================================
# C. Main Execution Block (Unchanged)
# =========================================================================

if __name__ == "__main__":
    # Ensure you have a folder named 'Vs' containing your CSV files.
    # The 'VsZ_15m_Smearing_Results.csv' will be saved inside 'Vs'.
    
    # You can change max_smear_m here if needed (e.g., from 5 to 10)
    output_csv_path = calculate_vsz_with_smearing(target_depth=15, max_smear_m=10, source_dir="Vs")
    
    # Proceed with analysis and plotting if calculation was successful
    if output_csv_path:
        # Note: You must have 'matplotlib' and 'seaborn' installed: 
        # pip install matplotlib seaborn
        analyze_and_plot_vsz_smearing_residuals(output_csv_path, target_depth=15, max_smear_m=10, source_dir="Vs")