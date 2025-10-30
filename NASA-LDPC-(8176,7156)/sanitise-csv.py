import csv
import os
import sys
PWD=os.path.dirname(os.path.realpath(__file__))
fileDir = PWD+'/report_3/5_specific_codeword_and_random_error_bit/'
input_file = fileDir+'32_minimum_sum_biterror-exact50on8176_count100000_dist_n8176m1022k7154_bit-error.csv'     # Replace with your input CSV filename
output_file = fileDir+'32_minimum_sum_biterror-exact50on8176_count100000_dist_n8176m1022k7154_bit-error-sanitized.csv'   # Replace with your desired output CSV filename


with open(input_file, 'r', encoding='utf-8') as infile:
    content = infile.read()

# Replace "\r\n " with " "
cleaned_content = content.replace('\n ', ' ')

with open(output_file, 'w', encoding='utf-8') as outfile:
    outfile.write(cleaned_content)

print("Replacement complete. Output written to output.csv")
