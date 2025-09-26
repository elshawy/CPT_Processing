import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
import scipy.interpolate
import scipy.stats

# Set style and context
sns.set_style("white")
plt.rc("axes.spines", top=False, right=False)
sns.set_context("paper")

# Smooth function for residuals
def smooth(x, y, xgrid):
    samples = np.random.choice(len(x), 50, replace=True)
    y_s = y[samples]
    x_s = x[samples]
    y_sm = sm_lowess(y_s, x_s, frac=1./3., it=5, return_sorted=False)
    y_grid = scipy.interpolate.interp1d(x_s, y_sm, fill_value='extrapolate')(xgrid)
    return y_grid

# Load dataset
df = pd.read_csv('Mode3_combined_results.csv')

# Extract the reference column (assuming Depth is the x-axis)
x_ax = df['Depth']

# Assuming 'Ic' is a column in your DataFrame for coloring
Ic_values = df['Ic']

# Define color categories based on the provided ranges for Ic
def get_marker_color(Ic_value):
    if Ic_value <= 1.31:
        return 'blue'
    elif Ic_value <= 2.05:
        return 'green'
    elif Ic_value <= 2.60:
        return 'orange'
    elif Ic_value <= 2.95:
        return 'purple'
    elif Ic_value <= 3.60:
        return 'cyan'
    else:
        return 'gray'

# Loop over the last 6 columns for plotting
for col in df.columns[-7:]:
    residuals = df[col].dropna()  # 빈칸 무시
    x_ax_nonan = x_ax.loc[residuals.index]  # 빈칸이 없는 residuals에 해당하는 x_ax
    Ic_values_nonan = Ic_values.loc[residuals.index]  # 빈칸이 없는 residuals에 해당하는 Ic_values
    
    # RMSE 계산
    rmse = np.sqrt(np.mean(residuals**2))
    print(f"{col} RMSE: {rmse:.3f}")
    
    xgrid = np.linspace(x_ax_nonan.min(), x_ax_nonan.max(), 200)
    K = 200
    smooths = np.stack([smooth(x_ax_nonan, residuals, xgrid) for k in range(K)]).T
    mean = np.nanmean(smooths, axis=1)
    stderr = np.nanstd(smooths, axis=1, ddof=1)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(xgrid, mean - 2 * stderr, color='w', linewidth=3, linestyle='-', alpha=0.8)
    plt.plot(xgrid, mean + 2 * stderr, color='w', linewidth=3, linestyle='-', alpha=0.8)
    plt.plot(xgrid, mean - 2 * stderr, color='k', linewidth=2, linestyle='--', alpha=0.8)
    plt.plot(xgrid, mean + 2 * stderr, color='k', linewidth=2, linestyle='--', alpha=0.8)

    plt.ylim(-3, 3)

    x_min, x_max = x_ax_nonan.min(), x_ax_nonan.max()
    buffer = (x_max - x_min) * 0.05
    plt.xlim(x_min - buffer, x_max + buffer)

    plt.grid(True, alpha=0.3)
    plt.yticks(fontsize=15)

    marker_colors = [get_marker_color(Ic_val) for Ic_val in Ic_values_nonan]
    plt.scatter(x_ax_nonan, residuals, s=15, c=marker_colors,edgecolor='k',linewidth=0.5, alpha=0.65)

    plt.plot(xgrid, mean, color='w', linestyle='-', linewidth=3)
    plt.plot(xgrid, mean, color='k', linestyle='-', linewidth=2, label=col)

    plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)

    plt.xlabel(x_ax.name, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Residuals', fontsize=15)

    # RMSE 텍스트 표시
    plt.text(0.8, 0.05, f'RMSE: {rmse:.3f}', transform=plt.gca().transAxes,
             fontsize=14, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.text(0.85, 0.9, f'{col}', fontsize=20, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'),
             horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)
    plt.text(0.01, 0.99, 'Underprediction', horizontalalignment='left', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=13, bbox=dict(facecolor='white', alpha=0.9, edgecolor='None', boxstyle='round,pad=0.2'))
    plt.text(0.01, 0.01, 'Overprediction', horizontalalignment='left', verticalalignment='bottom',
             transform=plt.gca().transAxes, fontsize=13, bbox=dict(facecolor='white', alpha=0.9, edgecolor='None', boxstyle='round,pad=0.2'))

    plt.savefig(f'Residuals_{col}.png', dpi=600)
    plt.close()
