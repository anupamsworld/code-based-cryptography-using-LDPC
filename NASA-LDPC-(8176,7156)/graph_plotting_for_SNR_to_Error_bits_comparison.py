# for certain snr value, plot the mean of error bits and meadian of error bits
# array of snr, mean of error bits, median of error bits are available

import os
import numpy as np
import matplotlib.pyplot as plt

snr_values = [
    17, 17.25, 17.5, 17.75, 18, 18.25, 18.29, 18.3, 18.5, 18.75,
    19, 19.25, 19.5, 19.75, 20, 20.25, 20.5, 20.75, 21, 21.25,
    21.5, 21.75, 22, 22.25, 22.5, 22.75
]

mean_bit_error = [
    195.042, 169.426, 145.492, 122.776, 102.804, 84.716, 82.439, 81.346, 68.6, 54.812,
    43.051, 33.042, 25.042, 17.864, 12.58, 8.83, 5.82, 3.673, 2.364, 1.298,
    0.655, 0.363, 0.15, 0.075, 0.029, 0.006
]

median_bit_error = [
    195, 169, 146, 123, 102, 85, 82, 81, 68, 55,
    43, 33, 25, 18, 12, 9, 6, 4, 2, 1,
    1, 0, 0, 0, 0, 0
]
#specify the gap between the points in the x-axis and y-axis
plt.figure(figsize=(9, 6))
plt.plot(snr_values, mean_bit_error, marker='o', label='Mean Bit Error', color='blue')
plt.plot(snr_values, median_bit_error, marker='x', label='Median Bit Error', color='orange')
#plt.title('Mean and Median Error Bits vs SNR')
plt.xlabel('SNR (dB)', fontsize=24)
plt.ylabel('Number of Error Bits', fontsize=24)
plt.xticks(np.arange(snr_values[0], snr_values[len(snr_values)-1],0.5), rotation=45)
plt.yticks(np.arange(0, 196, 20))
plt.xlim(16.8, 23)
plt.tick_params(axis='x', which='both', labelsize=18)
plt.tick_params(axis='y', which='both', labelsize=18)
plt.scatter([18.29, 18.29], [82.439, 81.346], color='red', s=100, marker='o', label='Mean, Median of max correctable Error bits', zorder=5)
plt.grid()
plt.legend()
plt.tight_layout()
# Set font size of the legend labels
plt.legend(fontsize=20)
PWD=os.path.dirname(os.path.realpath(__file__))
plt.savefig(PWD+'/snr_vs_bit_error.png')
plt.show()