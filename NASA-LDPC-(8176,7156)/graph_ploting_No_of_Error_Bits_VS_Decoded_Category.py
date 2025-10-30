import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

# Sample categories and values
'''
categories = ['A', 'B', 'C', 'D']
values1 = [10, 15, 7, 12]   # Dataset 1
values2 = [12, 18, 6, 10]   # Dataset 2
'''
'''
categories = [76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93]

values1 = [97, 86, 84, 91, 87, 84, 80, 75, 112, 80, 64, 56, 73, 58, 52, 46, 35, 29]
values2 = [0, 0, 0, 0, 4, 0, 2, 3, 6, 5, 10, 10, 18, 17, 26, 36, 46, 36]
'''

categories = [
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
    87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
    105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 119, 121
]

values1 = [
    4, 1, 2, 3, 2, 6, 9, 11, 8, 12, 20, 13, 18, 33, 25, 30, 35, 49, 37,
    56, 79, 70, 69, 86, 82, 81, 79, 97, 86, 84, 91, 87, 84, 80, 75, 112, 80, 64,
    56, 73, 58, 52, 46, 35, 29, 23, 9, 11, 5, 6, 3, 2, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0
]

values2 = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 2, 3, 6, 5, 10,
    10, 18, 17, 26, 36, 46, 36, 36, 47, 48, 54, 39, 30, 42, 32, 24, 24, 18,
    19, 19, 15, 6, 17, 12, 8, 4, 7, 5, 3, 1, 1, 2
]

error_bits_100000 = [10, 20, 30, 40, 50, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 90]

cat1Count_100000 = [100000, 100000, 100000, 100000, 100000, 99999, 99999, 99999, 99999, 99998, 99998, 99992, 99998, 99991, 99979, 99976, 99953, 99908, 99837, 99683, 99438, 99053, 98365, 97439, 66796]

cat3Count_100000 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 8, 2, 9, 3, 24, 47, 92, 163, 317, 562, 947, 1635, 2561, 33204]

print(np.arange(0, 76, 4))


# Set position of bar on X axis
x = np.arange(len(categories))  # label locations
width = 0.35  # width of the bars
plt.figure(figsize=(12, 6))
#plt.xticks(np.arange(49, 121, 3), rotation=45)
# Plotting the bars
plt.bar(x - width/2, values1, width, label='Decoded into Valid Code-Word', color='blue')
plt.bar(x + width/2, values2, width, label='Decoded into Invalid Code-Word', color='red')
plt.tick_params(axis='both', labelsize=18)
# Add labels and title
plt.xlabel('Number of Error bits in Code-Word', fontsize=24)
plt.ylabel('Number of Samples', fontsize=24)
#plt.title('Count of BP decoded categories for certain error bits')
#plt.xticks(x, categories)
plt.xticks(np.array([ 0,  4,  8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 69]), np.array([ 49,  53,  57,  61,  65,  69,  73,  77,  81,  85,  89,  93,  97, 101, 105, 109, 113, 119, 121]), rotation=45)
plt.grid()
plt.legend()
plt.tight_layout()
# Set font size of the legend labels
plt.legend(fontsize=20)
PWD=os.path.dirname(os.path.realpath(__file__))
plt.savefig(PWD+'/count_BP_decoded_categories_VS_error_bits.png')
# Show the plot
plt.show()


# Set position of bar on X axis
x = np.arange(len(error_bits_100000))  # label locations
width = 0.35  # width of the bars
plt.figure(figsize=(7,5))
#plt.xticks(np.arange(49, 121, 3), rotation=45)
# Plotting the bars
plt.bar(x, cat1Count_100000, color='blue')
#plt.tick_params(axis='x', labelsize=9)
plt.tick_params(axis='both', labelsize=9)
# Add labels and title
plt.xlabel('Number of Error bits in Code-Word', fontsize=13)
plt.ylabel('Samples decoded into Valid Code-Word', fontsize=13)
#plt.title('Count of BP decoded categories for certain error bits of category 1(100000 samples)')
plt.xticks(x, error_bits_100000)
#plt.xticks(np.array([ 0,  4,  8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 69]), np.array([ 49,  53,  57,  61,  65,  69,  73,  77,  81,  85,  89,  93,  97, 101, 105, 109, 113, 119, 121]), rotation=45)
plt.grid()
plt.legend()
PWD=os.path.dirname(os.path.realpath(__file__))
plt.savefig(PWD+'/count_BP_decoded_categories_VS_error_bits_category_1(100000 samples).png')
# Show the plot
plt.show()


# Set position of bar on X axis
x = np.arange(len(error_bits_100000))  # label locations
width = 0.35  # width of the bars
plt.figure(figsize=(6, 5))
#plt.xticks(np.arange(49, 121, 3), rotation=45)
# Plotting the bars
plt.bar(x, cat3Count_100000, color='red')
#plt.tick_params(axis='x', labelsize=9)
plt.tick_params(axis='both', labelsize=9)
# Add labels and title
plt.xlabel('Number of Error bits in Code-Word', fontsize=13)
plt.ylabel('Samples decoded into indeterminate Code-Word', fontsize=13)
#plt.title('Count of BP decoded categories for certain error bits of category 3(100000 samples)')
plt.xticks(x, error_bits_100000)
#plt.xticks(np.array([ 0,  4,  8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 69]), np.array([ 49,  53,  57,  61,  65,  69,  73,  77,  81,  85,  89,  93,  97, 101, 105, 109, 113, 119, 121]), rotation=45)
plt.grid()
plt.legend()
PWD=os.path.dirname(os.path.realpath(__file__))
plt.savefig(PWD+'/count_BP_decoded_categories_VS_error_bits_category_3(100000 samples).png')
# Show the plot
plt.show()