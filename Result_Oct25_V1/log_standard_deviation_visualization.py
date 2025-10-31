import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# CSV 로드
input_file = 'updated_combined_results.csv'
df = pd.read_csv(input_file)

# 마지막 열 (-1열): 로그 오차 (ln(Predicted) - ln(Measured))
log_errors = df.iloc[:, -1]

# 로그 표준편차 계산
log_std_dev = np.std(log_errors)
print(f"ln Std: {log_std_dev:.4f}")

# 1. 로그 오차 시계열 플롯
plt.figure(figsize=(10, 6))
plt.plot(df.index, log_errors, marker='o', linestyle='-', color='b', label='ln(Predicted) - ln(Measured)')
plt.axhline(0, color='r', linestyle='--', label='Zero Error Line')
plt.title('Log Residuals (ln(Predicted) - ln(Measured))')
plt.xlabel('Index')
plt.ylabel('Log Residual')
plt.legend()
plt.grid(True)
plt.savefig('Log_Residuals_Timeseries.png')
plt.show()

# 2. 히스토그램 + 정규분포 피팅
plt.figure(figsize=(10, 6))
sns.histplot(log_errors, kde=False, stat="density", bins=30, color='blue', label='Log Errors Histogram')

# 정규분포 곡선
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, np.mean(log_errors), np.std(log_errors))
plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution Fit')

# 평균 ± 표준편차 표시
mean = np.mean(log_errors)
std = np.std(log_errors)
plt.axvline(mean, color='g', linestyle='-', label='Mean')
plt.axvline(mean - std, color='r', linestyle='--', label=r'$\pm\sigma$')
plt.axvline(mean + std, color='r', linestyle='--')

# 빗금 표시
plt.fill_betweenx(p, mean - std, mean + std, color='gray', alpha=0.2, hatch='//')

# ln(σ) 텍스트
plt.text(xmin + (xmax - xmin) * 0.05, max(p) * 0.9,
         f'$\\ln(\\sigma)$: {log_std_dev:.4f}',
         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.title('Histogram and Normal Distribution of Log Residuals')
plt.xlabel('Log Residual')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.savefig('Log_Residuals_Histogram.png')
plt.show()
