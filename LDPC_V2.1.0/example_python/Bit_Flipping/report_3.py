# Reports are made by hardcoding the error_rate of BpDecoder irrespective of
# the actual error_rate, noOfErrorBits/n.
# New column is introduced, distance between codeword and decoded codeword

#new parity check matrix is taken from https://standards.nasa.gov/sites/default/files/standards/GSFC/A/0/gsfc-std-9100a-04-10-2017.pdf

def sum_lists_mod_2(lists):
  return np.array([sum(x) for x in zip(*lists)]) % 2



def getMinimumDistanceAndTheColumns(H,n):
    checkMinDistance=4
    checkMinDistance_MAX=4
    isMinDistanceFound = 0
    dependentColumns = []
    dependentColumnsIndices = []
    minDistCount=0
    while(checkMinDistance <= checkMinDistance_MAX):
        checkMinDistColCombs=itertools.combinations(range(n), r=checkMinDistance)
        count=0
        for checkMinDistColPattern in checkMinDistColCombs:
            #print()
            count+=1
            
            '''
            if count%500==0:
                print(f"count={count}")
            '''
            #print(f"checkMinDistColPattern={checkMinDistColPattern}")
            checkingColumns = []
            
            for column in checkMinDistColPattern:
                l1=[]
                for row in range(np.shape(H)[0]):
                    l1.append(H[row,column])
                checkingColumns.append(l1)
            columns=[]
            '''
            for i in range(len(checkingColumns)):
                columns.append(*checkingColumns[i])
            '''
            #print(f"checkingColumns={checkingColumns}")
                #print(f"")
            sumCol = sum_lists_mod_2(checkingColumns)
            #print(f"\nsumCol={sumCol}\tsum(sumCol)={sum(sumCol)}")
            if sum(sumCol) == 0:
                isMinDistanceFound = 1
                dependentColumns.append(checkingColumns)
                dependentColumnsIndices.append(list(checkMinDistColPattern))
                minDistCount+=1
                #print(f"\nDependent Columns=\n{dependentColumns}")
                #print(f"checkMinDistColPattern={checkMinDistColPattern}")
        print(f"count={count}")
        if isMinDistanceFound == 1:
            print(f"minDistCount={minDistCount}")
            print(f"dependentColumnsIndices=\n{dependentColumnsIndices}")
            return checkMinDistance, dependentColumnsIndices
        checkMinDistance += 1

    return isMinDistanceFound, dependentColumnsIndices

def vectorsAreIndependent(tempVectors):
    tempVectorsArray = np.array(tempVectors)
    rows, columns = np.shape(tempVectorsArray)
    if ldpc.mod2.rank(tempVectorsArray) == rows if rows < columns else columns:
        return 1
    else:
        return 0

def getIndependentVectors(n, dependentColumnsIndices, noOfVectorsNeeded, numberOfSetNeeded=1):
    if len(dependentColumnsIndices) < noOfVectorsNeeded:
        print(f"Number of set of independent columns can't be less than number of wanted indipendent vectors.")
    independentVectors = []
    allPossibleVectorPatterns = itertools.combinations(range(len(dependentColumnsIndices)),noOfVectorsNeeded)
    print(f"len(dependentColumnsIndices)={len(dependentColumnsIndices)}\tnoOfVectorsNeeded={noOfVectorsNeeded}")
    countIndependentVectors = 0
    countAllPossibleVectorPatterns = 0
    for vectorPattern in allPossibleVectorPatterns: # vectorPattern is nothinfg but the set of independent column indices of H
        countAllPossibleVectorPatterns += 1
        tempVectors = []
        for setOfPositions in vectorPattern:
            tempVector = [0]*n
            #print(f"")
            #print(f"dependentColumnsIndices[setOfPositions]={dependentColumnsIndices[setOfPositions]}")
            for indexFor1 in dependentColumnsIndices[setOfPositions]:
                
                tempVector[indexFor1] = 1
            tempVectors.append(tempVector)
        #print(f"\ntempVectors=\n{tempVectors}")
        #if sum(sum_lists_mod_2(tempVectors)) == 0:
        if vectorsAreIndependent(tempVectors) == 1 :
            independentVectors.append(tempVectors)
            countIndependentVectors += 1
            if numberOfSetNeeded != -1 and numberOfSetNeeded == countIndependentVectors:
                print(f"countAllPossibleVectorPatterns={countAllPossibleVectorPatterns}")
                return countIndependentVectors, independentVectors
    print(f"countAllPossibleVectorPatterns={countAllPossibleVectorPatterns}")
    return countIndependentVectors, independentVectors

def generate_random_permutation_matrix(n):
  """Generates a random permutation matrix of size NxN.

  Args:
    n: The size of the permutation matrix.

  Returns:
    A NumPy array representing the random permutation matrix.
  """

  permutation = np.random.permutation(n)
  matrix = np.zeros((n, n), dtype=int)
  for i, j in enumerate(permutation):
    matrix[i, j] = 1
  return matrix

import ldpc.mod2
from Hard_Message_Passing import *

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

def decode_All_Possible_Errors_With_HMP_and_BP_and_Generate_Report(H, n, k, m, noOfErrorBits, originalCodeword=None):
    #error_bit_permutations=itertools.permutations(range(n), r=e)
    error_bit_permutations=itertools.combinations(range(n), r=noOfErrorBits)
    #print(f"len(list(error_bit_permutations))={len(list(error_bit_permutations))}")
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
    error_rate=0.15
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
    fileName = PWD+'/report_3/1/'+bp_method+'_er'+str(error_rate)+'_dist_n'+str(n)+'m'+str(m)+'k'+str(k)+'_'+str(noOfErrorBits)+'bit-error.csv'
    
    headers = ['sent-message', 'received-message', 'HMP-decoded-value', 'HMP-syndrome', 'HMP-category', 'BP-decoded-value', 'BP-syndrome', 'BP-category', 'codeword_BP-Decoded']
    count=0
    import copy
    originalCodeword_copy = copy.deepcopy(originalCodeword)
    with open(fileName, mode='w', newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(headers)
        for error_bit_pattern in error_bit_permutations:
            
            if originalCodeword is None:
                temp_message = [0] * n # this is equivalent to received vector
            else:
                #temp_message = originalCodeword_copy
                temp_message = copy.deepcopy(originalCodeword_copy)
            #print(f"Type of temp_message={type(temp_message)}\ntemp_message={temp_message}")
            for error_bit_index in error_bit_pattern:
                temp_message[error_bit_index] ^= 1
            codeword_bits, check_bits = formTannerGraph(H)
            decodeWithHMP(codeword_bits, check_bits, temp_message, 1, noOfIteration_MAX)
            HMP_decoded_value = [codeword_bits[codewordNode]['bit'] for codewordNode in codeword_bits]
            HMP_syndrome = (HMP_decoded_value@H.T) % 2
            #print(f"shape of temp_message={np.array([temp_message]).shape}")
            BP_decoded_value = bpd.decode(np.array(temp_message))
            BP_syndrome = (BP_decoded_value@H.T) % 2
            #csvRow = [''.join(str(bit) for bit in [0]*n), ''.join(str(bit) for bit in temp_message), ''.join(str(bit) for bit in HMP_decoded_value), ''.join(str(bit) for bit in HMP_syndrome), getDecodedCategory(HMP_decoded_value, HMP_syndrome), ''.join(str(bit) for bit in BP_decoded_value), ''.join(str(bit) for bit in BP_syndrome), getDecodedCategory(BP_decoded_value, BP_syndrome)]
            #print(f"originalCodeword_copy={originalCodeword_copy}")
            #print(f"''.join(str(x) for x in originalCodeword_copy)={''.join(str(x) for x in originalCodeword_copy)}")
            csvRow = [''.join(map(str, originalCodeword_copy)), ''.join(map(str, temp_message)), ''.join(map(str, HMP_decoded_value)), ''.join(map(str, HMP_syndrome)), getDecodedCategory(originalCodeword_copy, HMP_decoded_value, HMP_syndrome), ''.join(map(str, BP_decoded_value)), ''.join(map(str, BP_syndrome)), getDecodedCategory(originalCodeword_copy, BP_decoded_value, BP_syndrome), hamming_distance(originalCodeword_copy, BP_decoded_value)]
            csvWriter.writerow(csvRow)
            count += 1

    print(f"count={count}")


def gf2_inverse(matrix):
    """Calculates the inverse of a matrix over GF(2).

    Args:
        matrix: A NumPy array representing the matrix.

    Returns:
        The inverse matrix, or None if the matrix is singular.
    """

    n = len(matrix)
    augmented_matrix = np.hstack((matrix, np.eye(n))).astype(int)
    for i in range(n):
        # Find the pivot row
        pivot_row = i
        while pivot_row < n and augmented_matrix[pivot_row, i] == 0:
            pivot_row += 1

        if pivot_row == n:
            return None  # Matrix is singular

        # Swap rows if necessary
        if pivot_row != i:
            augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]

        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            if augmented_matrix[j, i] == 1:
                augmented_matrix[j] ^= augmented_matrix[i]
    
    # Back substitution
    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            if augmented_matrix[j, i] == 1:
                augmented_matrix[j] ^= augmented_matrix[i]

    # Extract the inverse matrix
    inverse_matrix = augmented_matrix[:, n:]
    return inverse_matrix



# Example usage:
n = 25  # Code length
m = 10
k = n-m   # Number of information bits
#d_c = 3   # Variable node degree
#d_r = 4   # Check node degree

noOfErrorBits=2
noOfIteration_MAX = 10000


H=np.array(
[
[1,0,0,1,0, 1,1,0,0,0, 0,1,0,0,1, 0,1,1,0,0, 1,0,0,0,1],
[0,1,0,0,1, 0,1,1,0,0, 1,0,1,0,0, 0,0,1,1,0, 1,1,0,0,0],
[1,0,1,0,0, 0,0,1,1,0, 0,1,0,1,0, 0,0,0,1,1, 0,1,1,0,0],
[0,1,0,1,0, 0,0,0,1,1, 0,0,1,0,1, 1,0,0,0,1, 0,0,1,1,0],
[0,0,1,0,1, 1,0,0,0,1, 1,0,0,1,0, 1,1,0,0,0, 0,0,0,1,1],

[0,1,0,0,1, 0,0,0,1,1, 0,0,1,0,1, 1,0,1,0,0, 1,1,0,0,0],
[1,0,1,0,0, 1,0,0,0,1, 1,0,0,1,0, 0,1,0,1,0, 0,1,1,0,0],
[0,1,0,1,0, 1,1,0,0,0, 0,1,0,0,1, 0,0,1,0,1, 0,0,1,1,0],
[0,0,1,0,1, 0,1,1,0,0, 1,0,1,0,0, 1,0,0,1,0, 0,0,0,1,1],
[1,0,0,1,0, 0,0,1,1,0, 0,1,0,1,0, 0,1,0,0,1, 1,0,0,0,1]
]
)
#print(f"type of H={type(H)}")
import ldpc.code_util
print(f"ldpc.code_util.compute_exact_code_distance(H)={ldpc.code_util.compute_exact_code_distance(H)}")

#minDistanc, dependentColumnsIndices = getMinimumDistanceAndTheColumns(H,n)
#print(f"\nminDistanc={minDistanc}\ndependentColumns=\n{dependentColumns}")
#print(f"\nminDistanc={minDistanc}\ndependentColumnsIndices=\n{dependentColumnsIndices}")

#countIndependentVectors, independentVectors = getIndependentVectors(n, dependentColumnsIndices, 15, 1)
#print(f"countIndependentVectors={countIndependentVectors}")
#print(f"\nindependentVectors=\n{independentVectors}")


'''
G = np.array(
[[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
 [0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
 [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
 [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
 [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
) # This G is generated by calling ldpc.code_util.construct_generator_matrix(H) which is not of size 15x25
'''
G = np.array(
    [
[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
[1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
)
print(f"G=\n{G}")
print(f"G@H.T=\n{(G@H.T)%2}")

'''
G = ldpc.code_util.construct_generator_matrix(H)

print(f"G=\n{G.toarray()}")
print(f"G@H.T=\n{((G.toarray())@H.T)%2}")
'''
'''
rows, columns = np.shape(independentVectors[0])
print(f"np.shape(independentVectors[0])={np.shape(independentVectors[0])}")
found = 0
count = 0
for index, vector in enumerate(independentVectors):
    count += 1
    tempArray = np.array(independentVectors[index])
    #print(f"ldpc.mod2.rank(tempArray)={ldpc.mod2.rank(tempArray)}\tnp.shape(independentVectors[0])={np.shape(independentVectors[0])}")
    if ldpc.mod2.rank(tempArray) == rows if rows < columns else columns :
        print(f"independentVectors[{index}]=\n{independentVectors[index]}")
        found += 1
        break
print(f"count={count}\tfound={found}")
'''
#exit()

print(f"Rank of G={ldpc.mod2.rank(G)}")

#S = np.random.randint(2, size=(k,k))


S = np.array(
[[0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0,],
 [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,],
 [1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0,],
 [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1,],
 [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1,],
 [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1,],
 [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0,],
 [1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1,],
 [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0,],
 [1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,],
 [1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0,],
 [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,],
 [1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1,],
 [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1,],
 [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0,]]
)

# this is invertible

print(f"S=\n{S}")
S_inverse = []
print(f"rank(S)={ldpc.mod2.rank(S)}")
print(f"np.linalg.det(S)={np.linalg.det(S)}\tBut we need mod2 determinant")
if ldpc.mod2.rank(S) == k:
    print(f"S is invertible.")
    #S_inverse = np.linalg.inv(S).astype(int)
    S_inverse = gf2_inverse(S)

print(f"S_inverse=\n{S_inverse}")
print(f"(S@S_inverse)%2=\n{(S@S_inverse)%2}")

#P = generate_random_permutation_matrix(n)

P = np.array(
[[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,],
 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,],
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,],
 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]]
)

print(f"P=\n{P}")

P_inverse = gf2_inverse(P)
print(f"p_inverse=\n{P_inverse}")
print(f"(P@P_inverse)%2=\n{(P@P_inverse)%2}")

msg = np.array([1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1])


print(f"msg=\n{msg}")

msgS = (msg@S)%2
print(f"msgS=\n{msgS}")

msgSG = (msgS@G)%2
print(f"msgSG=\n{msgSG}")

msgG = (msg@G)%2
print(f"msgG=\n{msgG}")

decode_All_Possible_Errors_With_HMP_and_BP_and_Generate_Report(H, n, k, m, noOfErrorBits, msgG)


G_star = (((S @ G) % 2) @ P) % 2
print(f"G_star=\n{G_star}")

msgG_star = (msg@G_star) % 2
print(f"mG_star=\n{msgG_star}")

e = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
print(f"e=\n{e}")

y = (msgG_star + e)%2
print(f"y=\n{y}")

yp_inverse = (y@P_inverse) % 2
print(f"yp_inverse=\n{yp_inverse}")

from ldpc import BpDecoder
bpd = BpDecoder(
    #bpd = bp_decoder(
        H, #the parity check matrix
        error_rate=noOfErrorBits/n, # the error rate on each bit
        max_iter=noOfIteration_MAX, #the maximum iteration depth for BP
        bp_method="product_sum", #BP method. The other option is `minimum_sum'
    )

msgSG_decoded = bpd.decode(np.array(yp_inverse)) # mS is the BP_decoded_codeword
print(f"\nmsgSG_decoded=\n{msgSG_decoded}")
print(f"type(msgSG_decoded)={type(msgSG_decoded)}\t {np.shape(msgSG_decoded)}")
'''
msgSG_decoded = msgSG_decoded.reshape(1, 20)
print(f"\nmsgSG=\n{msgSG_decoded}")
print(f"type(msgSG)={type(msgSG_decoded)}\t {np.shape(msgSG_decoded)}")
'''
'''
messageCodewordTable = makeMessageCodewordTable(msgS, )
if sum((msgSG_decoded@H.T)%2) == 0:
    print(f"\n{msgSG_decoded} is a valid deoded codeword")
    getMessageFromCodeword(msgSG_decoded, messageCodewordTable)
'''
'''
decrypted_m = (mSG@S_inverse)
print(f"decrypted_m=\n{decrypted_m}")
'''