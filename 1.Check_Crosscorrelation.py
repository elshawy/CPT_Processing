import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from tqdm import tqdm
from scipy.signal import savgol_filter
from datetime import datetime 

# ==================================
# Description: This script processes CPT (Cone Penetration Test) raw data
#              to perform cross-correlation analysis between qc (cone resistance)
#              and fs (sleeve friction) signals. It identifies and corrects
#              depth shifts in the data, saves the corrected files, and
#              generates a comprehensive summary with timestamps.
# ==================================

# --------------------
# Global configuration
# --------------------
# Check for command-line argument for folder path
if len(sys.argv) > 1:
    folder_path = sys.argv[1]
else:
    print("\n[Usage] python 1.Check_Crosscorrelation.py <input_folder>\n")

    sys.exit(1)
    
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder_name = os.path.basename(os.path.normpath(folder_path)) + f"_CORRECTED_DATA_{timestamp}"
output_folder = os.path.join(os.getcwd(), output_folder_name)
# --------------------
# Paths & folders
# --------------------
initial_folder = os.path.join(output_folder, "Initial")
post_folder = os.path.join(output_folder, "Post_processing")

success_folder = os.path.join(post_folder, "CORRECTED_DATA_SUCCESS")
incomplete_folder = os.path.join(post_folder, "CORRECTED_DATA_INCOMPLETE")

fig_zero_init = os.path.join(initial_folder, "figures_zero")
fig_pos_init = os.path.join(initial_folder, "figures_positive")
fig_neg_init = os.path.join(initial_folder, "figures_negative")

fig_success_post = os.path.join(post_folder, "figures_success")
fig_incomplete_post = os.path.join(post_folder, "figures_incomplete")

for folder in [
    output_folder, initial_folder, post_folder, success_folder, incomplete_folder,
    fig_zero_init, fig_pos_init, fig_neg_init,
    fig_success_post, fig_incomplete_post
]:
    os.makedirs(folder, exist_ok=True)

csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

if not csv_files:
    print(f"No CSV files found in the directory: {folder_path}")
    sys.exit()

# --------------------
# Helper functions
# --------------------

def find_case_insensitive_column(df, column_names):
    """
    Finds a column in a DataFrame using a list of case-insensitive names.
    """
    for name in column_names:
        if name in df.columns:
            return name
    return None

def pearson_r_safe(x, y):
    """
    Calculates the Pearson correlation coefficient between two 1D arrays,
    handling cases with insufficient data or zero variance.
    """
    n = x.size
    if n < 5:
        return np.nan
    x_d = x - x.mean()
    y_d = y - y.mean()
    x_var = np.dot(x_d, x_d)
    y_var = np.dot(y_d, y_d)
    if x_var <= 0 or y_var <= 0:
        return np.nan
    return np.dot(x_d, y_d) / np.sqrt(x_var * y_var)

def step_shift_symmetric(depth, n):
    """
    Calculates the most common shift distance for a given number of steps (n).
    """
    if n == 0:
        return 0.0
    m = abs(n)
    if depth.size <= m:
        return np.nan
    diffs_m = depth[m:] - depth[:-m]
    unique_diffs, counts = np.unique(diffs_m, return_counts=True)
    if unique_diffs.size == 0:
        return np.nan
    mode_idx = np.argmax(counts)
    mode_value = unique_diffs[mode_idx]
    return np.sign(n) * mode_value

def run_crosscorr(depth, qc, fs, dz_median, max_steps=20, smoothing_window=5, search_range=0.2):
    """
    Performs a cross-correlation analysis between qc and fs data.
    """
    if smoothing_window <= 2: smoothing_window = 3
    if smoothing_window % 2 == 0: smoothing_window += 1
        
    qc_smoothed = savgol_filter(qc, window_length=smoothing_window, polyorder=2)
    fs_smoothed = savgol_filter(fs, window_length=smoothing_window, polyorder=2)

    n_values = np.arange(-max_steps, max_steps + 1)
    shifts = np.array([step_shift_symmetric(depth, n) for n in n_values])
    valid_mask = np.isfinite(shifts)
    if not np.any(valid_mask):
        return None

    n_values = n_values[valid_mask]
    shifts = shifts[valid_mask]

    correlations = []
    sample_counts = []
    for s in shifts:
        nstep = int(round(s / dz_median))
        qc_shifted = np.full_like(qc, np.nan)
        if nstep > 0:
            qc_shifted[nstep:] = qc[:-nstep]
        elif nstep < 0:
            qc_shifted[:nstep] = qc[-nstep:]
        else:
            qc_shifted = qc.copy()

        valid = np.isfinite(qc_shifted) & np.isfinite(fs_smoothed)
        sample_counts.append(valid.sum())
        if valid.sum() < 5:
            correlations.append(np.nan)
        else:
            correlations.append(pearson_r_safe(qc_shifted[valid], fs_smoothed[valid]))

    correlations = np.array(correlations)
    if np.all(~np.isfinite(correlations)):
        return None

    search_mask = np.abs(shifts) <= search_range
    filtered_shifts = shifts[search_mask]
    filtered_correlations = correlations[search_mask]
    filtered_samples = np.array(sample_counts)[search_mask]
    filtered_n_values = n_values[search_mask]

    if np.all(~np.isfinite(filtered_correlations)):
        return None

    max_idx = np.nanargmax(filtered_correlations)
    best_shift_final = filtered_shifts[max_idx]
    best_corr_final = filtered_correlations[max_idx]
    best_nstep_final = int(filtered_n_values[max_idx])
    best_n_samples_final = int(filtered_samples[max_idx])

    return {
        "best_shift": best_shift_final,
        "best_corr": best_corr_final,
        "best_nstep": best_nstep_final,
        "best_n_samples": best_n_samples_final,
        "shifts": shifts,
        "correlations": correlations,
        "n_values": n_values
    }


def fs_from_original(fs_data, total_qc_shift, dz_median):
    """
    Shifts the fs data array based on a total qc shift distance.
    """
    nstep = int(round(-total_qc_shift / dz_median))
    fs_corr = np.full_like(fs_data, np.nan)
    if nstep > 0:
        fs_corr[nstep:] = fs_data[:-nstep]
    elif nstep < 0:
        fs_corr[:nstep] = fs_data[-nstep:]
    else:
        fs_corr = fs_data.copy()
    return fs_corr, nstep * dz_median

def save_corr_plot(depth, qc, fs, ccf_result, fig_path, title):
    """
    Generates and saves a plot of the cross-correlation coefficient vs. shift distance.
    """
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(ccf_result["shifts"], ccf_result["correlations"], color='blue', linewidth=1.5, alpha=0.7)
    ax.axvline(ccf_result["best_shift"], color='k', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.2)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1.2)
    ax.plot(ccf_result["best_shift"], ccf_result["best_corr"], 'o', color='red')

    dz_median = np.median(np.diff(depth)) if depth.size > 1 else 0.01
    textstr = (
        f"Max r = {ccf_result['best_corr']:.3f}\n"
        f"Shift = {ccf_result['best_shift']:.3f} m\n"
        f"Lag step = {dz_median:.3f} m\n"
        f"N = {ccf_result['best_n_samples']}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    ax.set_title(title, pad=20)
    ax.set_xlabel("qc shift distance [m]")
    ax.set_ylabel("Cross-Correlation coefficient")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(-dz_median * 20, dz_median * 20)
    ax.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

# --------------------
# Main loop
# --------------------
summary_rows = []

for file in tqdm(csv_files, desc="Processing files"):
    try:
        df = pd.read_csv(file)
    except Exception as e:
        print(f"Error reading file {os.path.basename(file)}: {e}")
        continue
    
    depth_col = find_case_insensitive_column(df, ["Depth", "depth", "DEPTH"])
    qc_col = find_case_insensitive_column(df, ["qc", "Qc", "QC"])
    fs_col = find_case_insensitive_column(df, ["fs", "Fs", "FS"])
    u_col = find_case_insensitive_column(df, ["u", "U"])

    if not all([depth_col, qc_col, fs_col]):
        print(f"Skipping {os.path.basename(file)}: Missing required columns 'Depth', 'qc', or 'fs'.")
        continue

    depth0 = df[depth_col].to_numpy()
    qc0 = df[qc_col].to_numpy()
    fs0 = df[fs_col].to_numpy()
    u0 = df[u_col].to_numpy() if u_col else None
    
    mask_all = np.isfinite(depth0) & np.isfinite(qc0) & np.isfinite(fs0)
    if u_col:
        mask_all &= np.isfinite(u0)
    
    if np.sum(mask_all) == 0:
        print(f"Skipping {os.path.basename(file)}: No valid data points found.")
        continue

    depth0, qc0, fs0 = depth0[mask_all], qc0[mask_all], fs0[mask_all]
    if u_col:
        u0 = u0[mask_all]

    nonzero_mask = depth0 != 0
    depth0, qc0, fs0 = depth0[nonzero_mask], qc0[nonzero_mask], fs0[nonzero_mask]
    if u_col:
        u0 = u0[nonzero_mask]
        
    order = np.argsort(depth0)
    depth0, qc0, fs0 = depth0[order], qc0[order], fs0[order]
    if u_col:
        u0 = u0[order]

    uniq_depths, idx_start = np.unique(depth0, return_index=True)
    if uniq_depths.size != depth0.size:
        grouped = {}
        for d, q, f, *other_cols in zip(depth0, qc0, fs0, *([u0] if u_col else [[]])):
            grouped.setdefault(d, [[], [], []])
            grouped[d][0].append(q)
            grouped[d][1].append(f)
            if u_col:
                grouped[d][2].append(other_cols[0])
        depth0 = np.array(sorted(grouped.keys()))
        qc0 = np.array([np.mean(grouped[d][0]) for d in depth0])
        fs0 = np.array([np.mean(grouped[d][1]) for d in depth0])
        if u_col:
            u0 = np.array([np.mean(grouped[d][2]) for d in depth0])

    if depth0.size < 10:
        print(f"Skipping {os.path.basename(file)}: Not enough data points (< 10).")
        continue

    dz_median = np.median(np.diff(depth0)) if depth0.size > 1 else np.nan

    initial_best = run_crosscorr(depth0, qc0, fs0, dz_median, max_steps=50)
    if initial_best is None:
        print(f"Skipping {os.path.basename(file)}: Cross-correlation analysis failed.")
        continue

    if initial_best["best_shift"] < 0:
        shift_type_init = "Negative"
        fig_folder_init = fig_neg_init
    elif initial_best["best_shift"] == 0:
        shift_type_init = "Zero"
        fig_folder_init = fig_zero_init
    else:
        shift_type_init = "Positive"
        fig_folder_init = fig_pos_init

    base = os.path.splitext(os.path.basename(file))[0]
    fig_path_init = os.path.join(fig_folder_init, f"{base}_initial_corr.png")
    save_corr_plot(depth0, qc0, fs0, initial_best, fig_path_init, f"{base} - Initial")

    current_fs = fs0.copy()
    cumulative_qc_shift = 0.0
    final_best_for_plot = None
    shift_type_post = ""
    corrected_file_path = ""
    note_post = ""
    total_iterations = 0

    if initial_best["best_shift"] > 0:
        for i in range(100):
            total_iterations = i + 1
            ccf_result = run_crosscorr(depth0, qc0, current_fs, dz_median, max_steps=20)
            
            if ccf_result is None or ccf_result["best_n_samples"] < 5:
                note_post = f"Correction failed after {i} iterations"
                break

            if abs(ccf_result["best_shift"]) < dz_median:
                final_best_for_plot = ccf_result
                cumulative_qc_shift += ccf_result["best_shift"]
                note_post = f"Positive shift corrected successfully in {total_iterations} iterations"
                break

            shift_to_apply = ccf_result["best_shift"]
            cumulative_qc_shift += shift_to_apply
            current_fs, _ = fs_from_original(fs0, cumulative_qc_shift, dz_median)
            final_best_for_plot = ccf_result

        if final_best_for_plot is not None and abs(final_best_for_plot["best_shift"]) < dz_median:
            shift_type_post = "Positive_success"
            save_folder = success_folder
            note_post = note_post if note_post else f"Positive shift corrected successfully in {total_iterations} iterations"
        else:
            shift_type_post = "Positive_incomplete"
            save_folder = incomplete_folder
            note_post = note_post if note_post else f"Residual shift remains after {total_iterations} iterations (shift={final_best_for_plot['best_shift']:.3f} m)"

        final_fs = current_fs

        corrected_data = {"Depth": depth0, "qc_orig": qc0, "fs_corrected": final_fs}
        if u_col:
            corrected_data["u_orig"] = u0
        corrected_df = pd.DataFrame(corrected_data)
        corrected_file_path = os.path.join(save_folder, f"{base}_fs_corrected.csv")
        corrected_df.to_csv(corrected_file_path, index=False)

        fig_folder_post = fig_success_post if shift_type_post == "Positive_success" else fig_incomplete_post
        fig_path_post = os.path.join(fig_folder_post, f"{base}_post_corr.png")
        if final_best_for_plot is not None:
            save_corr_plot(depth0, qc0, final_fs, final_best_for_plot, fig_path_post, f"{base} - Post-processing")
        else:
            fig_path_post = "NaN"

    else:
        final_fs = fs0.copy()
        cumulative_qc_shift = 0.0
        corrected_file_path = "Not corrected"
        fig_folder_post = None
        note_post = "No correction needed"
        shift_type_post = shift_type_init
        final_best_for_plot = initial_best
        fig_path_post = "NaN"

    summary_rows.append({
        "file": os.path.basename(file),
        "median_dz": round(dz_median, 3),
        "total_N": depth0.size,
        "final_total_qc_shift_m": round(cumulative_qc_shift, 3),
        "final_fs_shift_applied_m": round(-cumulative_qc_shift, 3),
        "final_best_residual_shift_m": round(final_best_for_plot["best_shift"], 3) if final_best_for_plot else np.nan,
        "final_best_residual_nstep": final_best_for_plot["best_nstep"] if final_best_for_plot else np.nan,
        "final_best_corr": round(final_best_for_plot["best_corr"], 3) if final_best_for_plot else np.nan,
        "used_N": final_best_for_plot["best_n_samples"] if final_best_for_plot else np.nan,
        "Initial_shift_type": shift_type_init,
        "Post_shift_type": shift_type_post,
        "figure_path_initial": fig_path_init,
        "figure_path_post": fig_path_post,
        "corrected_data_path": corrected_file_path,
        "Note_post": note_post
    })

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_folder, "summary.csv"), index=False)
