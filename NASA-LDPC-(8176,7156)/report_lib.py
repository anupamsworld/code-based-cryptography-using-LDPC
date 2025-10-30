def hamming_distance(vector1, vector2):
  """Calculates the Hamming distance between two binary vectors.

  Args:
    vector1: The first binary vector as a list of 0s and 1s.
    vector2: The second binary vector as a list of 0s and 1s.

  Returns:
    The Hamming distance between the two vectors.
  """

  if len(vector1) != len(vector2):
    raise ValueError("Vectors must have the same length.")

  distance = 0
  for bit1, bit2 in zip(vector1, vector2):
    if bit1 != bit2:
      distance += 1

  return distance

def decode_All_Possible_Errors_With_HMP_and_BP_and_Generate_Report_3(H, n, k, m, noOfErrorBits, noOfIteration_MAX, fileDir, originalCodeword=None):
    import itertools
    import sys
    import os
    import numpy as np
    import lib.GPU_Computations as GPUC
    import cupy
    PWD=os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.abspath(PWD+'/../'))
    import Bit_Flipping.Hard_Message_Passing as HMP

    if originalCodeword is None:
        print(f"originalCodeword is None")
    else:
        print(f"originalCodeword={originalCodeword}")
    #print(f"e/n={noOfErrorBits/n}")
    
    import csv
    from ldpc import BpDecoder
    #from ldpc import bp_decoder
    #from ldpc import bp_decoder
    #from ldpc import *
    #import ldpc
    import os
    PWD=os.path.dirname(os.path.realpath(__file__))

    bp_method="minimum_sum"
    #bp_method = "product_sum"
    #schedule = "serial"
    schedule = "parallel"    
    

    print(f"noOfIteration_MAX={noOfIteration_MAX}")

    reportNumber = "7"
    fileName = fileDir+reportNumber+"_"+bp_method+'_biterror-prob0.00110078277886497064579256360078'+'_dist_n'+str(n)+'m'+str(m)+'k'+str(k)+'_'+'bit-error.csv'
    fileName2 = fileDir+reportNumber+"_BP-step-by-step.csv"
    #headers = ['sent-message', 'received-message', 'HMP-decoded-value', 'HMP-syndrome', 'HMP-category', 'BP-decoded-value', 'BP-syndrome', 'BP-category', 'codeword_BP-Decoded']
    headers = ['total-error-bits','codeword-error-indexes', 'BP-decoded-error-indexes', 'BP-iterations', 'HMP-category','BP-category', 'codeword_BP-Decoded']
    headers2 = ['seq-no', 'codeword', 'hamming-distance-from-previous', 'BP-category']
    count=10
    import copy
    if originalCodeword is None:
        originalCodeword_copy = [0] * n
    else:
        originalCodeword_copy = copy.deepcopy(originalCodeword)
    with open(fileName, mode='a', newline='') as csvFile, open(fileName2, mode='a', newline='') as csvBPbreakeup:
        csvWriter = csv.writer(csvFile)
        csvBPbreakeup_Writer = csv.writer(csvBPbreakeup)
        csvWriter.writerow(headers)
        csvBPbreakeup_Writer.writerow(headers2)

        error_positions_set = set()
        
        index = 0
        indexes=[]
        binary_array=[0]*n
        for sample_index in range(count):
            print(f"")
            #if count > 18:
            
            temp_message = copy.deepcopy(originalCodeword_copy)
            #print(f"Type of temp_message={type(temp_message)}\ntemp_message={temp_message}")

            # create error matrix
            
            while True:
              e, error_positions = GPUC.GF2makeErrorMatrixUsingGPU(n, method=1)
              #print(f"type(e)= {type(e)}")
              #print(f"type(error_positions)= {type(error_positions)}")
              if str(error_positions) not in error_positions_set:
                 error_positions_set.add(str(error_positions))
                 ###print(f"Found new error combination. Breaking.")
                 break
            
            '''
            while True:
              if index % 9 == 0 :
                  binary_array[count] = 1
                  indexes.append(count)
                  break
              else:
                 index += 1
            '''

            ###print(f"e is made.")
            #sum_e = sum(e)
            sum_e = cupy.sum(e)
            print(f"Number of 1's in the error vector(sum_e)= {sum_e}")
            error_rate = sum_e/n
            ###print(f"After forming error matrix with randomized bits, the error rate becomes= {error_rate}")
            

            # infuse error in codeword
            #temp_message = cupy.bitwise_xor(np.array(temp_message), np.array(e))
            #print(f"type(temp_message)= {type(temp_message)}")
            sum_temp_message = cupy.sum(cupy.asarray(temp_message))
            print(f"Number of 1's in the temp_message(sum_temp_message)= {sum_temp_message}")
            temp_message = cupy.asnumpy(cupy.bitwise_xor(cupy.array(temp_message), cupy.array(e)))
            sum_temp_message = cupy.sum(cupy.asarray(temp_message))
            print(f"After pushing error, Number of 1's in the temp_message(sum_temp_message)= {sum_temp_message}")
            #print(f"type(temp_message)= {type(temp_message)}")
            temp_message = list(temp_message)
            #print(f"type(temp_message)= {type(temp_message)}")



            '''
            codeword_bits, check_bits = HMP.formTannerGraph(H)
            print(f"Starting HMP decoding.")
            HMP.decodeWithHMP(codeword_bits, check_bits, temp_message, 1, noOfIteration_MAX)
            print(f"Completed HMP decoding.")
            HMP_decoded_value = [codeword_bits[codewordNode]['bit'] for codewordNode in codeword_bits]
            #HMP_syndrome = (HMP_decoded_value@H.T) % 2
            HMP_syndrome = GPUC.GF2MatrixMultUsingGpu(HMP_decoded_value, H.T)
            '''


            #print(f"shape of temp_message={np.array([temp_message]).shape}")

            bpd = BpDecoder(
            #bpd = bp_decoder(
                H, #the parity check matrix
                max_iter=noOfIteration_MAX,
                error_rate=float(error_rate), # the error rate on each bit
                input_vector_type="received_vector",
                schedule = schedule # the BP schedule
            )
            BP_Decoded_category = -1
            print(f"Started BP decoding.")
            for iterarion_index in range(noOfIteration_MAX):
                BP_decoded_value = bpd.decode(np.array(temp_message))
                #BP_syndrome = (BP_decoded_value@H.T) % 2
                BP_syndrome = GPUC.GF2MatrixMultUsingGpu(BP_decoded_value, H.T)
                BP_Decoded_category = HMP.getDecodedCategory(originalCodeword_copy, BP_decoded_value, BP_syndrome)
                csvBPbreakeup_Writer.writerow([iterarion_index+1, cupy.where(cupy.asarray(BP_decoded_value)==1)[0], hamming_distance(temp_message, BP_decoded_value), BP_Decoded_category])
                temp_message = copy.deepcopy(BP_decoded_value)

                if BP_Decoded_category == 1 or BP_Decoded_category == 2:
                   break
                
            
            
            
            #csvRow = [str(error_positions), HMP.getDecodedCategory(originalCodeword_copy, HMP_decoded_value, HMP_syndrome), HMP.getDecodedCategory(originalCodeword_copy, BP_decoded_value, BP_syndrome), hamming_distance(originalCodeword_copy, BP_decoded_value)]
            
            csvRow = [sum_e, error_positions, cupy.where(cupy.asarray(BP_decoded_value)==1)[0], bpd.iter,  "-", BP_Decoded_category, hamming_distance(originalCodeword_copy, BP_decoded_value)]

            csvWriter.writerow(csvRow)
                        
            print(f"sample_index={sample_index}")
            
            

    print(f"sample_index={sample_index}")

def decode_All_Possible_Errors_With_HMP_and_BP_and_Generate_Report_2(H, n, k, m, noOfErrorBits, noOfIteration_MAX, fileDir, originalCodeword=None, specialAnalysis=False):
    import itertools
    import sys
    import os
    import numpy as np
    import lib.GPU_Computations as GPUC
    import cupy
    PWD=os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.abspath(PWD+'/../'))
    import Bit_Flipping.Hard_Message_Passing as HMP

    if originalCodeword is None:
        print(f"originalCodeword is None")
    else:
        #print(f"originalCodeword={originalCodeword}")
        print(f"originalCodeword is not None")
    #print(f"e/n={noOfErrorBits/n}")
    
    import csv
    from ldpc import BpDecoder
    #from ldpc import bp_decoder
    #from ldpc import bp_decoder
    #from ldpc import *
    #import ldpc
    
    bp_method="minimum_sum" #default
    #bp_method="product_sum"
    print(f"noOfIteration_MAX={noOfIteration_MAX}")
    
    np.set_printoptions(threshold=np.inf)

    import os
    PWD=os.path.dirname(os.path.realpath(__file__))
    fileName = fileDir+"32_"+bp_method+'_biterror-exact50on8176_count100000'+'_dist_n'+str(n)+'m'+str(m)+'k'+str(k)+'_'+'bit-error.csv'
    
    #headers = ['sent-message', 'received-message', 'HMP-decoded-value', 'HMP-syndrome', 'HMP-category', 'BP-decoded-value', 'BP-syndrome', 'BP-category', 'codeword_BP-Decoded']
    #headers = ['total-error-bits','codeword-error-indexes', 'BP-decoded-error-indexes', 'BP-iterations', 'HMP-category','BP-category', 'codeword_BP-Decoded']
    headers = ['total_error_bits','codeword_error_indexes', 'BP_decoded_error_indexes', 'BP_iterations', 'HMP_category','BP_category', 'codeword_BP_Decoded']

    if specialAnalysis:
        count=1
    else:
        count= 2589
    
    import copy
    if originalCodeword is None:
        originalCodeword_copy = [0] * n
    else:
        originalCodeword_copy = copy.deepcopy(originalCodeword)
    with open(fileName, mode='a', newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(headers)
        error_positions_set = set()
        
        index = 0
        indexes=[]
        binary_array=[0]*n
        for sample_index in range(count):
            print(f"")
            #if count > 18:
            
            temp_message = copy.deepcopy(originalCodeword_copy)
            #print(f"Type of temp_message={type(temp_message)}\ntemp_message={temp_message}")

            # create error matrix
            
            #print(f"Finding new error matrix.")
            while True:
                e, error_positions = GPUC.GF2makeErrorMatrixUsingGPU(n, noOfErrorBits=noOfErrorBits, method=1)
                sum_e = cupy.sum(e)
                #print(f"type(e)= {type(e)}")
                #print(f"type(error_positions)= {type(error_positions)}")
                hasToBreak = False
                if str(error_positions) not in error_positions_set:
                    if noOfErrorBits != -1:
                        if noOfErrorBits == sum_e:
                            ###print(f"Found new error combination with {noOfErrorBits} bits. Breaking.")
                            hasToBreak = True
                    else:
                        hasToBreak = True
                if hasToBreak:
                    error_positions_set.add(str(error_positions))
                    #print(f"Found new error matrix. Breaking.")
                    break
              
            
            '''
            while True:
              if index % 9 == 0 :
                  binary_array[count] = 1
                  indexes.append(count)
                  break
              else:
                 index += 1
            '''
            
            #print(f"Number of 1's in the error vector(sum_e)= {sum_e}")
            error_rate = sum_e/n
            ###print(f"After forming error matrix with randomized bits, the error rate becomes= {error_rate}")
            

            # infuse error in codeword
            #temp_message = cupy.bitwise_xor(np.array(temp_message), np.array(e))
            #print(f"type(temp_message)= {type(temp_message)}")
            sum_temp_message = cupy.sum(cupy.asarray(temp_message))
            #print(f"Number of 1's in the temp_message(sum_temp_message)= {sum_temp_message}")
            temp_message = cupy.asnumpy(cupy.bitwise_xor(cupy.array(temp_message), cupy.array(e)))
            sum_temp_message = cupy.sum(cupy.asarray(temp_message))
            #print(f"After pushing error, Number of 1's in the temp_message(sum_temp_message)= {sum_temp_message}")
            #print(f"type(temp_message)= {type(temp_message)}")
            temp_message = list(temp_message)
            #print(f"type(temp_message)= {type(temp_message)}")



            '''
            codeword_bits, check_bits = HMP.formTannerGraph(H)
            print(f"Starting HMP decoding.")
            HMP.decodeWithHMP(codeword_bits, check_bits, temp_message, 1, noOfIteration_MAX)
            print(f"Completed HMP decoding.")
            HMP_decoded_value = [codeword_bits[codewordNode]['bit'] for codewordNode in codeword_bits]
            #HMP_syndrome = (HMP_decoded_value@H.T) % 2
            HMP_syndrome = GPUC.GF2MatrixMultUsingGpu(HMP_decoded_value, H.T)
            '''


            #print(f"shape of temp_message={np.array([temp_message]).shape}")

            bpd = BpDecoder(
            #bpd = bp_decoder(
                H, #the parity check matrix
                max_iter=noOfIteration_MAX,
                error_rate=float(error_rate), # the error rate on each bit
                input_vector_type="received_vector"
                #,
                #schedule = "serial" # the BP schedule
                
            )
            ''', #the parity check matrix
                error_rate=float(error_rate), # the error rate on each bit
                max_iter=noOfIteration_MAX, #the maximum iteration depth for BP
                bp_method=bp_method, #BP method. The other option is 'minimum_sum'
                '''
            print(f"Started BP decoding.")
            BP_decoded_value = bpd.decode(np.array(temp_message))
            print(f"Completed BP decoding.")
            #BP_syndrome = (BP_decoded_value@H.T) % 2
            BP_syndrome = GPUC.GF2MatrixMultUsingGpu(BP_decoded_value, H.T)
            #csvRow = [str(error_positions), HMP.getDecodedCategory(originalCodeword_copy, HMP_decoded_value, HMP_syndrome), HMP.getDecodedCategory(originalCodeword_copy, BP_decoded_value, BP_syndrome), hamming_distance(originalCodeword_copy, BP_decoded_value)]
            
            #csvRow = [sum_e, error_positions, cupy.where(cupy.asarray(BP_decoded_value)==1)[0], bpd.iter,  HMP.getDecodedCategory(originalCodeword_copy, HMP_decoded_value, HMP_syndrome), HMP.getDecodedCategory(originalCodeword_copy, BP_decoded_value, BP_syndrome), hamming_distance(originalCodeword_copy, BP_decoded_value)]
            BP_Decoded_Category = HMP.getDecodedCategory(originalCodeword_copy, BP_decoded_value, BP_syndrome)
            csvRow = [sum_e, error_positions, cupy.where(cupy.bitwise_xor(cupy.asarray(BP_decoded_value), cupy.asarray(e))==1)[0] if (BP_Decoded_Category != 1) else "[-]", bpd.iter,  "-", BP_Decoded_Category, hamming_distance(originalCodeword_copy, BP_decoded_value)]

            csvWriter.writerow(csvRow)
                        
            print(f"sample_index={sample_index}")
            
            

    print(f"sample_index={sample_index}")



def decode_All_Possible_Errors_With_HMP_and_BP_and_Generate_Report(H, n, k, m, noOfErrorBits, noOfIteration_MAX, fileDir, originalCodeword=None):
    import itertools
    import sys
    import os
    import numpy as np
    import lib.GPU_Computations as GPUC
    PWD=os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.abspath(PWD+'/../'))
    import Bit_Flipping.Hard_Message_Passing as HMP
    #error_bit_permutations=itertools.permutations(range(n), r=e)
    error_bit_combinations=itertools.combinations(range(n), r=noOfErrorBits)
    print(f"len(list(error_bit_permutations))={len(list(error_bit_combinations))}")
    #print(f"error_bit_permutations=\n{list(error_bit_permutations)}")
    if originalCodeword is None:
        print(f"originalCodeword is None")
    else:
        print(f"originalCodeword={originalCodeword}")
    print(f"e/n={noOfErrorBits/n}")
    
    import csv
    from ldpc import BpDecoder
    #from ldpc import bp_decoder
    #from ldpc import bp_decoder
    #from ldpc import *
    #import ldpc
    #error_rate=0.15
    error_rate=noOfErrorBits / n
    bp_method="minimum_sum"
    #bp_method="product_sum"
    print(f"noOfIteration_MAX={noOfIteration_MAX}")
    bpd = BpDecoder(
    #bpd = bp_decoder(
        H, #the parity check matrix
        error_rate=error_rate, # the error rate on each bit
        max_iter=noOfIteration_MAX, #the maximum iteration depth for BP
        bp_method=bp_method, #BP method. The other option is 'minimum_sum'
    )
    import os
    PWD=os.path.dirname(os.path.realpath(__file__))
    fileName = fileDir+bp_method+'_er'+str(error_rate)+'_dist_n'+str(n)+'m'+str(m)+'k'+str(k)+'_'+str(noOfErrorBits)+'bit-error_from73584.csv'
    
    #headers = ['sent-message', 'received-message', 'HMP-decoded-value', 'HMP-syndrome', 'HMP-category', 'BP-decoded-value', 'BP-syndrome', 'BP-category', 'codeword_BP-Decoded']
    headers = ['error-indexes', 'HMP-category','BP-category', 'codeword_BP-Decoded']

    count=0
    import copy
    if originalCodeword is None:
        originalCodeword_copy = [0] * n
    else:
        originalCodeword_copy = copy.deepcopy(originalCodeword)
    with open(fileName, mode='a', newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(headers)
        for error_bit_pattern in error_bit_combinations:
            print(f"hello")
            if count > 73584: #8176*9 for 9 bit errors I wanted to avoid burst errors
              print(f"Hi")
              temp_message = copy.deepcopy(originalCodeword_copy)
              #print(f"Type of temp_message={type(temp_message)}\ntemp_message={temp_message}")
              for error_bit_index in error_bit_pattern:
                  temp_message[error_bit_index] ^= 1
              codeword_bits, check_bits = HMP.formTannerGraph(H)
              HMP.decodeWithHMP(codeword_bits, check_bits, temp_message, 1, noOfIteration_MAX)
              HMP_decoded_value = [codeword_bits[codewordNode]['bit'] for codewordNode in codeword_bits]
              #HMP_syndrome = (HMP_decoded_value@H.T) % 2
              HMP_syndrome = GPUC.GF2MatrixMultUsingGpu(HMP_decoded_value, H.T)
              #print(f"shape of temp_message={np.array([temp_message]).shape}")
              BP_decoded_value = bpd.decode(np.array(temp_message))
              #BP_syndrome = (BP_decoded_value@H.T) % 2
              BP_syndrome = GPUC.GF2MatrixMultUsingGpu(BP_decoded_value, H.T)
              #csvRow = [''.join(str(bit) for bit in [0]*n), ''.join(str(bit) for bit in temp_message), ''.join(str(bit) for bit in HMP_decoded_value), ''.join(str(bit) for bit in HMP_syndrome), getDecodedCategory(HMP_decoded_value, HMP_syndrome), ''.join(str(bit) for bit in BP_decoded_value), ''.join(str(bit) for bit in BP_syndrome), getDecodedCategory(BP_decoded_value, BP_syndrome)]
              #print(f"originalCodeword_copy={originalCodeword_copy}")
              #print(f"''.join(str(x) for x in originalCodeword_copy)={''.join(str(x) for x in originalCodeword_copy)}")
              csvRow = [str(error_bit_pattern), HMP.getDecodedCategory(originalCodeword_copy, HMP_decoded_value, HMP_syndrome), HMP.getDecodedCategory(originalCodeword_copy, BP_decoded_value, BP_syndrome), hamming_distance(originalCodeword_copy, BP_decoded_value)]
              csvWriter.writerow(csvRow)
              
              print(f"count={count}")
              count += 1
            
            
            if count % 50 == 0:
               print(f"count(round)={count}")
               return
            

    print(f"count={count}")




def sameMessageTroubleReport(n, k, pad_length, messages_list:list, messageG_star_list:list, error_list:list, full_file_path:str):
    import csv
    import cupy
    import numpy as np
    import sys
    headers = ['raw-message-bit-index','pad-index', 'messageG_star-index', 'error-index']
    with open(full_file_path, mode='a', newline="") as csvFile:

        print(csv.field_size_limit())
        #csv.field_size_limit(sys.maxsize)
        #print(csv.field_size_limit())

        csvWriter = csv.writer(csvFile, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(headers)
        for messageSample_index in range(len(messages_list)):
            raw_message_bit_index = cupy.where(cupy.asarray((messages_list[messageSample_index])[:k-pad_length])==1)[0]
            pad_index = cupy.where(cupy.asarray((messages_list[messageSample_index])[k-pad_length:])==1)[0]
            messageG_star_index = cupy.where(cupy.asarray(messageG_star_list[messageSample_index])==1)[0]
            error_index = cupy.where(cupy.asarray(error_list[messageSample_index])==1)[0]

            csvRow = [str(raw_message_bit_index), str(pad_index), np.array2string(messageG_star_index, threshold=np.inf), str(error_index)]
            csvWriter.writerow(csvRow)

    None