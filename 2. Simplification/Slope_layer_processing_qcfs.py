import os
import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import sys

def geometric_mean(series):
    """
    Calculates the geometric mean of a pandas Series.
    This function is used to find a robust average for non-negative,
    positively skewed data, which is common in geotechnical parameters.

    Args:
        series (pd.Series): The pandas Series containing the data.

    Returns:
        float: The calculated geometric mean. Returns NaN if the series is empty
               or contains non-positive values.
    """
    # Filter out non-positive values to avoid errors with the logarithm.
    series = series[series > 0]
    if series.empty:
        return np.nan
    # Calculate geometric mean using the logarithmic formula.
    return np.exp(np.log(series).mean())


def process_csv_file(file_path, output_folder, min_thickness_m):
    """
    Processes a single CSV file containing CPT data to detect distinct layers,
    plot the results, and calculate statistical properties for each layer.
    The Ic value is calculated based on the provided CPT class logic.

    Args:
        file_path (str): The full path to the input CSV file.
        output_folder (str): The folder where output plots and statistics will be saved.
        min_thickness_m (float): The minimum thickness in meters for a detected layer.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # -----------------------
    # Validate and get required columns (Qc, Fs, U)
    # -----------------------
    df_cols = [col.lower() for col in df.columns]

    # Find Qc or qc column
    if 'qc' in df_cols:
        qc_col = df.columns[df_cols.index('qc')]
        qc = df[qc_col].values
    else:
        print(f"Required column 'Qc' or 'qc' not found in {file_path}. Skipping.")
        return

    # Find Fs or fs column, default to zeros if not found
    if 'fs' in df_cols:
        fs_col = df.columns[df_cols.index('fs')]
        fs = df[fs_col].values
    else:
        print(f"Required column 'Fs' or 'fs' not found in {file_path}. Using Fs = 0.")
        fs = np.zeros(len(df))

    # Find U or u column, default to zeros if not found
    if 'u' in df_cols:
        u_col = df.columns[df_cols.index('u')]
        u = df[u_col].values
    else:
        print(f"Required column 'U' or 'u' not found in {file_path}. Using U = 0.")
        u = np.zeros(len(df))

    depth = df['Depth'].values

    # -----------------------
    # Calculate Ic based on provided CPT class logic
    # -----------------------
    pa = 0.1  # atmospheric pressure (MPa)
    ground_water_level = 1.0
    net_area_ratio = 0.8

    qt = qc + u * (1 - net_area_ratio)

    # Simplified gamma calculation
    default_gamma = 0.00981 * 1.9
    gamma_w = 9.80665
    
    # Corrected Rf calculation to prevent log errors
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

    # Create a temporary DataFrame to handle NaN values before change point detection
    temp_df = pd.DataFrame({'Depth': depth, 'Qc': qc, 'Ic': Ic})
    temp_df.dropna(subset=['Ic'], inplace=True)

    if temp_df.empty:
        print(f"Warning: All calculated Ic values for {file_path} are invalid. Skipping layer analysis.")
        return

    # Standardize Qc and the newly filtered Ic for Change Point Detection
    data_for_segmentation = StandardScaler().fit_transform(temp_df[['Qc', 'Ic']].values)

    # Determine the minimum layer size in data points based on a minimum thickness in meters.
    depth_interval = np.median(np.diff(temp_df['Depth'].values))
    if depth_interval == 0 or np.isnan(depth_interval):
        print(f"Could not determine depth interval for {file_path}. Skipping min_thickness setting.")
        min_size = 0.5
    else:
        min_size = max(1, int(min_thickness_m / depth_interval))

    print(f"Using min_size of {min_size} data points (approx. {min_thickness_m} meters).")

    # -----------------------
    # Perform Change Point Detection
    # -----------------------
    algo = rpt.Pelt(model="l2", jump=1, min_size=min_size).fit(data_for_segmentation)
    result = algo.predict(pen=5)
    
    # Ruptures returns the length of the dataset as the final breakpoint, which is
    # an out-of-bounds index for pandas. We remove it to prevent an IndexError.
    if result and result[-1] == len(data_for_segmentation):
        result = result[:-1]

    # Map the detected change points back to the original DataFrame's indices using Depth
    layer_depths = temp_df.iloc[result]['Depth'].values
    layer_indices = sorted([df[df['Depth'] == d].index[0] for d in layer_depths])

    # Ensure the change point list includes the start (index 0) and end of the dataset.
    if 0 not in layer_indices:
        layer_indices.insert(0, 0)
    if len(df) not in layer_indices:
        layer_indices.append(len(df))

    # -----------------------
    # Plotting
    # -----------------------
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
    plot_path = os.path.join(output_folder, f"{file_name_prefix}_layers_plot_qc_Ic.png")
    plt.savefig(plot_path)
    plt.close(fig)

    print(f"Layer plot saved to {plot_path}")

    # -----------------------
    # Calculate and save layer statistics
    # -----------------------
    stats_data = []
    df['Ic_calculated'] = Ic # Temporarily add calculated Ic to the dataframe for easy access
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
    stats_file_path = os.path.join(output_folder, f"{file_name_prefix}_layers_stats_Qc_Ic.csv")
    stats_df.to_csv(stats_file_path, index=False)

    print(f"Layer statistics saved to {stats_file_path}")


def main():
    """
    Main execution function to find and process all CSV files in a user-specified directory.
    """
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    else:
        print("Error: Please provide the path to the folder containing CSV files as a command-line argument.")
        print("Usage: python cpt_layer_analysis.py <path_to_folder>")
        sys.exit(1)

    if not os.path.isdir(input_folder):
        print(f"Error: The specified folder '{input_folder}' does not exist.")
        return

    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    min_thickness_m = 1

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_main_folder = os.path.join(os.getcwd(), f"Simplification_results_{timestamp}")
    os.makedirs(output_main_folder, exist_ok=True)

    if not csv_files:
        print("No CSV files found in the specified directory.")
        return

    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        file_path = os.path.join(input_folder, csv_file)
        process_csv_file(file_path, output_main_folder, min_thickness_m)
        print("-" * 50)


if __name__ == "__main__":
    main()
