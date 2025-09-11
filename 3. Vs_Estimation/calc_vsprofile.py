import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
from vs_calc import CPT, VsProfile, vs30_correlations

# -----------------------------
# Command-line Arguments
# -----------------------------
if len(sys.argv) < 2:
    print("Usage: python asd.py <input_folder>")
    sys.exit(1)

# Take input folder from command-line argument
input_dir = Path(sys.argv[1])
if not input_dir.exists() or not input_dir.is_dir():
    print(f"Error: {input_dir} is not a valid directory.")
    sys.exit(1)

# Timestamp string for output naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create output folder: "<input_folder_name>_results_<timestamp>"
out_dir = Path(f'./{input_dir.name}_results_{timestamp}')
out_dir.mkdir(exist_ok=True)

# Error log file with timestamp in filename
log_file = out_dir / f"error_log_{timestamp}.txt"

# -----------------------------
# Main Processing Loop
# -----------------------------
for file_path in input_dir.glob('*.csv'):
    file_name = file_path.stem  # Get file name without extension
    try:
        # -----------------------------
        # Load CPT Data
        # -----------------------------
        cpt = CPT.from_file(str(file_path))

        # -----------------------------
        # Define Vs Correlation Models
        # -----------------------------
        vs_correlations = "andrus_2007_pleistocene"  # Andrus et al. (2007), Pleistocene soils
        vs_correlations2 = "mcgann_2015"             # McGann et al. (2015)
        vs_correlations3 = "mcgann_2018"             # McGann et al. (2018)

        # -----------------------------
        # Create VsProfile Objects
        # -----------------------------
        vs_profile = VsProfile.from_cpt(cpt, vs_correlations)
        vs_profile2 = VsProfile.from_cpt(cpt, vs_correlations2)
        vs_profile3 = VsProfile.from_cpt(cpt, vs_correlations3)

        # Extract computed Vs values
        depth, vs = vs_profile.depth, vs_profile.vs
        depth2, vs2 = vs_profile2.depth, vs_profile2.vs
        depth3, vs3 = vs_profile3.depth, vs_profile3.vs

        # -----------------------------
        # Load Original CPT CSV
        # -----------------------------
        df = pd.read_csv(file_path)
        # Round depth values to 2 decimal places for alignment
        df.iloc[:, 0] = df.iloc[:, 0].round(2)

        # -----------------------------
        # Add Vs Correlation Results
        # -----------------------------
        vs_df = pd.DataFrame({
            'Depth': depth,
            'Andrus-P Vs': vs,
            'Mc15 Vs': vs2,
            'Mc18 Vs': vs3
        })
        df = pd.merge(df, vs_df, how='left', on='Depth')

        # -----------------------------
        # Add Soil Behavior Index (Ic)
        # -----------------------------
        ic_df = pd.DataFrame({
            'Depth': cpt.depth,
            'Ic': cpt.Ic
        })

        # Merge using nearest match on Depth
        df = pd.merge_asof(df.sort_values('Depth'),
                           ic_df.sort_values('Depth'),
                           on='Depth')
        df.iloc[:, 0] = df.iloc[:, 0].round(2)

        # -----------------------------
        # Save Final Result
        # -----------------------------
        df.to_csv(out_dir / f'{file_name}_result.csv', index=False)
        print(f'Done: {file_name}')

    except Exception as e:
        # -----------------------------
        # Error Logging with Timestamp
        # -----------------------------
        error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"[{error_time}] Error in file {file_name}: {str(e)}\n")
        print(f"Skipped {file_name} due to error.")
