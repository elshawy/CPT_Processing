# ===============================================================================
# CPT Data Cross-Correlation and Shift Correction
#
# This script processes raw CPT (Cone Penetration Test) data files (CSV format)
# to identify and correct depth shifts between the cone resistance (qc) and
# sleeve friction (fs) signals.
#
# The primary steps are:
# 1.  Read CPT data from a specified input folder.
# 2.  Perform an initial cross-correlation analysis between qc and fs to detect
#     any depth shift.
# 3.  Categorize files based on the detected shift (positive, negative, or zero).
# 4.  If a positive shift is found, the fs signal is corrected by shifting it
#     to align with the qc signal.
# 5.  A post-correction cross-correlation is performed to verify the alignment.
# 6.  The script organizes the original and corrected data, along with
#     diagnostic plots, into a structured output directory.
# 7.  A summary report (summary.csv) is generated to document the process
#     and results for each file.
#
# Author: JHKim
# Version: 1.2 (Added dedicated folder for positive shift data)
# Date: September 23, 2025
# ===============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import shutil
from tqdm import tqdm
from scipy.signal import savgol_filter
from datetime import datetime

# -------------------------------------------------------------------------------
# Project Configuration & Directory Setup
# -------------------------------------------------------------------------------

# Check for command-line argument for the input folder path
if len(sys.argv) > 1:
    folder_path = sys.argv[1]
else:
    print("\n[Usage] python 1.Check_Crosscorrelation.py <input_folder_path>\n")
    sys.exit(1)

# Generate a unique timestamp for the output folder to avoid overwriting previous runs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder_name = os.path.basename(os.path.normpath(folder_path)) + f"_CORRECTED_DATA_{timestamp}"
output_folder = os.path.join(os.getcwd(), output_folder_name)

# Define subfolders for organizing output data and figures
initial_folder = os.path.join(output_folder, "Initial")
post_folder = os.path.join(output_folder, "Post_processing")

# Subfolders for original data, categorized by initial shift
data_zero_init = os.path.join(initial_folder, "data_zero")
data_neg_init = os.path.join(initial_folder, "data_negative")
data_pos_init = os.path.join(initial_folder, "data_positive") # New folder for positive shift data

# Subfolders for corrected data, categorized by correction success
success_folder = os.path.join(post_folder, "CORRECTED_DATA_SUCCESS")
incomplete_folder = os.path.join(post_folder, "CORRECTED_DATA_INCOMPLETE")

# Subfolders for diagnostic plots, categorized by shift type and stage
fig_zero_init = os.path.join(initial_folder, "figures_zero")
fig_pos_init = os.path.join(initial_folder, "figures_positive")
fig_neg_init = os.path.join(initial_folder, "figures_negative")
fig_success_post = os.path.join(post_folder, "figures_success")
fig_incomplete_post = os.path.join(post_folder, "figures_incomplete")

# Create all necessary directories if they don't exist
for folder in [
    initial_folder, post_folder, data_zero_init, data_neg_init, data_pos_init,
    success_folder, incomplete_folder,
    fig_zero_init, fig_pos_init, fig_neg_init,
    fig_success_post, fig_incomplete_post
]:
    os.makedirs(folder, exist_ok=True)

# Find all CSV files in the user-specified input folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# -------------------------------------------------------------------------------
# Core Functions for Cross-Correlation and Data Handling
# -------------------------------------------------------------------------------

def sample_crosscov_and_corr(qc, fs, max_lag_steps):
    """
    Computes the sample cross-covariance and cross-correlation coefficients.

    Args:
        qc (np.array): Array of cone resistance data.
        fs (np.array): Array of sleeve friction data.
        max_lag_steps (int): The maximum number of steps (data points) to lag
                             the fs signal relative to the qc signal.

    Returns:
        tuple: A tuple containing:
               - lags (np.array): Array of lag steps.
               - c_vals (np.array): Array of cross-covariance values.
               - r_vals (np.array): Array of cross-correlation coefficient values.
    """
    qc = np.asarray(qc, dtype=float)
    fs = np.asarray(fs, dtype=float)

    n = qc.size
    if n != fs.size or n < 5:
        return None, None, None

    qbar = np.nanmean(qc)
    fbar = np.nanmean(fs)

    lags = np.arange(-max_lag_steps, max_lag_steps + 1)
    c_vals = np.full(lags.shape, np.nan)

    for idx, k in enumerate(lags):
        if k >= 0:
            if n - k <= 0:
                continue
            a = qc[:n - k] - qbar
            b = fs[k:] - fbar
            c_vals[idx] = np.nansum(a * b) / n
        else:
            m = -k
            if n - m <= 0:
                continue
            a = fs[:n - m] - fbar
            b = qc[m:] - qbar
            c_vals[idx] = np.nansum(a * b) / n

    c0_qc = np.nansum((qc - qbar) ** 2) / n
    c0_fs = np.nansum((fs - fbar) ** 2) / n
    denom = np.sqrt(c0_qc * c0_fs)

    r_vals = c_vals / denom if denom > 0 else np.full_like(c_vals, np.nan)
    return lags, c_vals, r_vals

def run_crosscorr(depth, qc, fs, dz_median, max_steps, smoothing_window=None):
    """
    Executes the cross-correlation analysis, finds the optimal shift,
    and calculates the best correlation coefficient.

    Args:
        depth (np.array): Depth data.
        qc (np.array): Cone resistance data.
        fs (np.array): Sleeve friction data.
        dz_median (float): Median depth interval.
        max_steps (int): Maximum lag steps for the cross-correlation.
        smoothing_window (int, optional): Window size for Savitzky-Golay
                                          smoothing. Defaults to None.

    Returns:
        dict: A dictionary containing the best shift, correlation, and lag,
              or None if the analysis fails.
    """
    if smoothing_window is not None and smoothing_window > 2:
        if smoothing_window % 2 == 0:
            smoothing_window += 1
        qc_smooth = savgol_filter(qc, window_length=smoothing_window, polyorder=2)
        fs_smooth = savgol_filter(fs, window_length=smoothing_window, polyorder=2)
    else:
        qc_smooth = qc
        fs_smooth = fs

    lags, c_vals, r_vals = sample_crosscov_and_corr(qc_smooth, fs_smooth, max_steps)
    if lags is None or np.all(~np.isfinite(r_vals)):
        return None

    shifts = lags * dz_median
    max_idx = np.nanargmax(r_vals)

    best_shift = shifts[max_idx]
    best_corr = r_vals[max_idx]
    best_lag = lags[max_idx]

    return {
        "best_shift": best_shift,
        "best_corr": best_corr,
        "best_lag": best_lag,
        "lags": lags,
        "shifts": shifts,
        "r_vals": r_vals
    }

def fs_from_original(fs_data, total_qc_shift, dz_median):
    """
    Realigns the sleeve friction (fs) data by applying a depth shift.

    Args:
        fs_data (np.array): Original sleeve friction data.
        total_qc_shift (float): The total depth shift to apply (in meters).
        dz_median (float): Median depth interval.

    Returns:
        tuple: A tuple containing the corrected fs data array and the
               actual applied shift in meters.
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

def save_corr_plot(ccf_result, fig_path, title):
    """
    Generates and saves a plot of the cross-correlation curve.

    Args:
        ccf_result (dict): Dictionary containing cross-correlation results.
        fig_path (str): The full path to save the figure file.
        title (str): Title for the plot.
    """
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(ccf_result["shifts"], ccf_result["r_vals"], color='blue', linewidth=1.5, alpha=0.7)
    ax.axvline(ccf_result["best_shift"], color='k', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.2)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1.2)
    ax.plot(ccf_result["best_shift"], ccf_result["best_corr"], 'o', color='red')

    textstr = (
        f"Max r = {ccf_result['best_corr']:.3f}\n"
        f"Shift = {ccf_result['best_shift']:.3f} m\n"
        f"Lag step = {ccf_result['best_lag']}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    ax.set_title(title, pad=20)
    ax.set_xlabel("fs shift distance [m]")
    ax.set_ylabel("Cross-Correlation coefficient")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

def save_comparison_plot(depth, qc_orig, fs_orig, fs_corrected, fig_path, title):
    """
    Creates and saves a side-by-side plot comparing original and corrected data.

    Args:
        depth (np.array): Depth data.
        qc_orig (np.array): Original qc data.
        fs_orig (np.array): Original fs data.
        fs_corrected (np.array): Corrected fs data.
        fig_path (str): The full path to save the figure file.
        title (str): Title for the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Plot original data
    ax1.plot(qc_orig, depth, label='qc (original)', color='red', alpha=0.7)
    ax1.plot(fs_orig, depth, label='fs (original)', color='blue', alpha=0.7)
    ax1.set_title(f"Original Data: {title}")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Depth (m)")
    ax1.legend()
    ax1.invert_yaxis()
    ax1.grid(True, linestyle=":")

    # Plot corrected data
    ax2.plot(qc_orig, depth, label='qc (original)', color='red', alpha=0.7)
    ax2.plot(fs_corrected, depth, label='fs (corrected)', color='green', alpha=0.7)
    ax2.set_title(f"Corrected Data: {title}")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Depth (m)")
    ax2.legend()
    ax2.invert_yaxis()
    ax2.grid(True, linestyle=":")

    plt.tight_layout()
    #plt.savefig(fig_path, dpi=300)
    plt.close()

# -------------------------------------------------------------------------------
# Main Processing Loop
# -------------------------------------------------------------------------------

summary_rows = []

# Loop through all CSV files in the input folder with a progress bar
for file in tqdm(csv_files, desc="Processing files"):
    try:
        df = pd.read_csv(file)
        # Convert all column names to lowercase for case-insensitivity
        df.columns = df.columns.str.lower()
    except Exception as e:
        print(f"Skipping {file} due to read error: {e}")
        continue

    required_cols = ["depth", "qc", "fs"]
    has_u_col = "u" in df.columns
    if not all(col in df.columns for col in required_cols):
        print(f"Skipping {file}: Missing one or more required columns (Depth, qc, fs).")
        continue

    base = os.path.splitext(os.path.basename(file))[0]

    depth0 = df["depth"].to_numpy()
    qc0 = df["qc"].to_numpy()
    fs0 = df["fs"].to_numpy()
    u0 = df["u"].to_numpy() if has_u_col else None

    # Filter out non-finite and non-positive data points
    mask_all = np.isfinite(depth0) & np.isfinite(qc0) & np.isfinite(fs0)
    if has_u_col:
        mask_all &= np.isfinite(u0)
    positive_values_mask = (qc0 >= 0) & (fs0 >= 0)
    mask_all &= positive_values_mask

    depth0, qc0, fs0 = depth0[mask_all], qc0[mask_all], fs0[mask_all]
    if has_u_col:
        u0 = u0[mask_all]

    # Filter out zero depth points
    nonzero_mask = depth0 != 0
    depth0, qc0, fs0 = depth0[nonzero_mask], qc0[nonzero_mask], fs0[nonzero_mask]
    if has_u_col:
        u0 = u0[nonzero_mask]

    # Ensure there's enough data to perform analysis
    if depth0.size < 10:
        print(f"Skipping {file}: Not enough valid data points (< 10).")
        continue

    # Calculate median depth interval (dz)
    dz_median = np.median(np.diff(depth0)) if depth0.size > 1 else np.nan
    if np.isnan(dz_median) or dz_median == 0:
        print(f"Skipping {file}: Invalid depth interval (dz).")
        continue

    # --- Step 1: Initial Cross-Correlation and File Classification ---
    # Perform initial correlation to find the natural shift (15 steps)
    initial_best = run_crosscorr(depth0, qc0, fs0, dz_median, max_steps=15)
    if initial_best is None:
        print(f"Skipping {file}: Initial cross-correlation failed.")
        continue

    initial_qc_shift = initial_best["best_shift"]

    # Categorize and copy the original file to its corresponding folder
    if initial_qc_shift < 0:
        fig_folder_init = fig_neg_init
        shutil.copy(file, os.path.join(data_neg_init, os.path.basename(file)))
    elif initial_qc_shift == 0:
        fig_folder_init = fig_zero_init
        shutil.copy(file, os.path.join(data_zero_init, os.path.basename(file)))
    else:  # initial_qc_shift > 0
        fig_folder_init = fig_pos_init
        shutil.copy(file, os.path.join(data_pos_init, os.path.basename(file))) # Changed to save to a separate data folder

    # Save the initial CCF plot (using a wider range of 100 steps for visualization)
    initial_plot_result = run_crosscorr(depth0, qc0, fs0, dz_median, max_steps=100)
    if initial_plot_result:
        fig_path_init = os.path.join(fig_folder_init, f"{base}_initial_corr.png")
        save_corr_plot(initial_plot_result, fig_path_init, f"{base} - Initial CCF")
    else:
        fig_path_init = "N/A"

    # --- Step 2: Conditional Correction for Positive Shifts Only ---
    # Apply a correction only if the initial shift is positive
    if initial_qc_shift > 0:
        # Shift the fs data to align with qc
        final_fs, final_applied_shift = fs_from_original(fs0, initial_qc_shift, dz_median)

        # Recalculate CCF on the corrected data to check for remaining residual shift
        post_corr_result = run_crosscorr(depth0, qc0, final_fs, dz_median, max_steps=15)

        # Determine if the correction was successful (residual shift is negligible)
        if post_corr_result and abs(post_corr_result["best_shift"]) < (dz_median / 2):
            shift_type = "Positive_shift_SUCCESS"
            note = "Positive initial shift. Correction applied. Post-corr shift is negligible."
            save_folder = success_folder
            fig_folder_post = fig_success_post
        else:
            shift_type = "Positive_shift_INCOMPLETE"
            note = "Positive initial shift. Correction applied but post-corr shift is still significant."
            save_folder = incomplete_folder
            fig_folder_post = fig_incomplete_post

        # Save the corrected data file
        corrected_data = {"Depth": depth0, "qc": qc0, "fs_corrected": final_fs}
        if has_u_col:
            corrected_data["u"] = u0
        corrected_df = pd.DataFrame(corrected_data)
        corrected_file_path = os.path.join(save_folder, f"{base}_fs_corrected.csv")
        corrected_df.to_csv(corrected_file_path, index=False)

        # Plot the post-correction CCF and a comparison plot
        post_plot_result = run_crosscorr(depth0, qc0, final_fs, dz_median, max_steps=100)
        if post_plot_result:
            post_corr_fig_path = os.path.join(fig_folder_post, f"{base}_post_corr.png")
            save_corr_plot(post_plot_result, post_corr_fig_path, f"{base} - Post-Correction CCF")
        else:
            post_corr_fig_path = "N/A"

        comp_fig_path = os.path.join(fig_folder_post, f"{base}_comparison_plot.png")
        save_comparison_plot(depth0, qc0, fs0, final_fs, comp_fig_path, base)

        summary_info = {
            "final_fs_applied_shift_m": round(final_applied_shift, 3),
            "post_corr_residual_shift_m": round(post_corr_result["best_shift"], 3) if post_corr_result else None,
            "post_corr_final_corr": round(post_corr_result["best_corr"], 3) if post_corr_result else None,
            "figure_path_post_ccf": post_corr_fig_path,
            "figure_path_comparison": comp_fig_path,
            "corrected_data_path": corrected_file_path
        }
    else:  # No correction is performed for negative or zero shifts
        shift_type = "Negative_or_Zero_INCOMPLETE"
        note = "No correction attempted due to negative or zero initial shift."

        # Set paths to the original file for the summary
        corrected_file_path = file
        post_corr_fig_path = "N/A"
        comp_fig_path = "N/A"

        summary_info = {
            "final_fs_applied_shift_m": 0.0,
            "post_corr_residual_shift_m": initial_best["best_shift"],
            "post_corr_final_corr": initial_best["best_corr"],
            "figure_path_post_ccf": "N/A",
            "figure_path_comparison": "N/A",
            "corrected_data_path": corrected_file_path
        }

    # Append results for the current file to the summary list
    summary_rows.append({
        "file": os.path.basename(file),
        "median_dz": round(dz_median, 3),
        "total_N": depth0.size,
        "initial_best_shift_m": round(initial_qc_shift, 3),
        "initial_best_corr": round(initial_best["best_corr"], 3),
        **summary_info,
        "correction_type": shift_type,
        "figure_path_initial_ccf": fig_path_init,
        "Note": note
    })

# Export the comprehensive summary table to a CSV file
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_folder, "summary.csv"), index=False)
