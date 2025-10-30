import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
PWD=os.path.dirname(os.path.realpath(__file__))
# Replace with your file path and column name
csv_file = PWD+'/80_error_bits_cat1_iteration_freq_report.csv'
column_name = 'bp_iterations'

# Load CSV and extract the column
df = pd.read_csv(csv_file)
data = df[column_name].dropna()  # Remove NaN values

# Frequency of each unique value
frequencies = data.value_counts().sort_index()

# Sort values
sorted_data = frequencies.index.to_list()
#print(f"\nSorted data: {sorted_data}")
#print(f"\nFrequencies: {frequencies}")


# Calculate mean and standard deviation
mean_val = data.mean()
std_val = data.std()
median_val = data.median()
mode_val = data.mode()[0]  # Get the first mode if there are multiple

# Plotting
plt.figure(figsize=(10, 6))
#plt.bar(frequencies.index, frequencies.values, color='skyblue', edgecolor='black')
plt.plot(frequencies.index, frequencies.values, marker='o', linestyle='-', color='blue', linewidth=3)
# Set fewer x-ticks (e.g., every 5th tick)
all_ticks = list(frequencies.index.astype(str))
step = max(1, len(all_ticks) // 100)  # Show at most 10 ticks
#print(f"\nAll ticks: {all_ticks}, \nStep: {step}")
#plt.xticks(ticks=np.arange(0, len(all_ticks), step), labels=[all_ticks[i] for i in range(0, len(all_ticks), step)], rotation=45)
tick_indices = np.arange(sorted_data[0], len(sorted_data), step)
#tick_values = [sorted_data[i] for i in range(len(tick_indices))]
tick_values = np.arange(sorted_data[0], len(sorted_data), step)
#print(f"\nTicks: {tick_indices}")
#print(f"\nTick values: {tick_values}")
#labels = [sorted_data[i] for i in range(len(ticks))]
#print(f"\nLabels: {labels}")
plt.xticks(ticks=tick_indices, labels=tick_values, rotation=45)


# Set x-axis limits
plt.xlim(left=0, right=50)

# Add mean and standard deviation lines
plt.axvline(x=mean_val, color='red', linestyle='--', label=f'Mean = {mean_val:.2f}')
plt.axvline(x=median_val, color='blue', linestyle='--', label=f'Median = {median_val:.2f}')
plt.axvline(x=mode_val, color='black', linestyle='--', label=f'Mode = {mode_val:.2f}')
#plt.axvline(x=mean_val + std_val, color='green', linestyle='--', label=f'Mean + 1σ = {mean_val + std_val:.2f}')
#plt.axvline(x=mean_val - std_val, color='orange', linestyle='--', label=f'Mean - 1σ = {mean_val - std_val:.2f}')



# Labels and title
plt.xlabel('BP Iterations', fontsize=24)
plt.ylabel('Iteration Frequency', fontsize=24)
#plt.title(f'Frequency of values in "{column_name}"\nMean = {mean_val:.2f}, Std Dev = {std_val:.2f}')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Set font size of the legend labels
plt.legend(fontsize=20)

# Set font size of the x and y labels
plt.tick_params(axis='both', labelsize=18)  # Both X and Y ticks
plt.subplots_adjust(left=0.150)
plt.savefig(PWD+'/iteration_frequency_vs_iteration.png')
plt.show()




# Data
x = [10, 20, 30, 40, 50, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 90]
y = [1.5018, 2.0349, 2.5952, 3.2376, 4.0533, 5.9393, 6.1474, 6.3289, 6.5594, 6.8021,
     7.0209, 7.3773, 7.7007, 7.9914, 8.4120, 9.2612, 10.0551, 11.0126, 12.9651, 14.6738,
     16.2507, 18.8883, 25.2424, 29.8925, 75.6033]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', color='b', linewidth=3)
plt.xlabel('Error Bits', fontsize=24)
plt.ylabel('Average BP Iterations', fontsize=24)
plt.grid(True)
plt.tight_layout()

# Set font size of the x and y labels
plt.tick_params(axis='both', labelsize=14)  # Both X and Y ticks

plt.savefig(PWD+'/error_bits_vs_average_bp_iterations.png')
plt.show()
