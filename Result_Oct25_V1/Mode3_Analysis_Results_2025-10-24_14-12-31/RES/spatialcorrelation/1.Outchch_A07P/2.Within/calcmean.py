# Python
import pandas as pd

# Load your DataFrame
df = pd.read_csv('Mode3_combined_results.csv')  # Replace with your file path

# Group by 'File Key_2' and calculate the mean for the last 7 columns
mean_columns = df.iloc[:, -1:].groupby(df['File Key']).transform('mean')

# Add the mean columns to the original DataFrame
mean_columns = mean_columns.add_prefix('Mean_')
df = pd.concat([df, mean_columns], axis=1)

# Save the updated DataFrame to a new file
df.to_csv('updated_combined_results.csv', index=False)  # Replace with your desired output file path
