import numpy as np
import cupy

def generate_gallager_ldpc_code(n, k, d_v, d_c):
  """
  Generates a random Gallager LDPC code.

  Args:
    n: Code length.
    k: Number of information bits.
    d_v: Variable node degree.
    d_c: Check node degree.

  Returns:
    A numpy array representing the parity-check matrix H.
  """

  H = np.zeros((n - k, n), dtype=int)

  # Initialize a list to keep track of the degree of each variable node
  variable_degrees = np.zeros(n, dtype=int)
  print(f"variable_degrees={variable_degrees}")

  # Iterate over each row of the parity-check matrix
  for i in range(n - k):
    # Select d_c variable nodes randomly
    #selected_nodes = np.random.choice(n, d_c, replace=False)
    #selected_nodes = np.random.Generator.integers(n,size=d_c)
    selected_nodes = np.random.default_rng().integers(n,size=d_c)
    print(selected_nodes)
    #input("Press enter")
    # Ensure that the degree of each selected variable node is less than d_v
    while any(variable_degrees[selected_nodes] >= d_v) or set(selected_nodes).__len__()!=d_c:
      #selected_nodes = np.random.choice(n, d_c, replace=False)
      #selected_nodes = np.random.Generator.integers(n,size=d_c)
      selected_nodes = np.random.default_rng().integers(n,size=d_c)
      print(f"set(selected_nodes).__len__={set(selected_nodes).__len__()}")
      print(f"\n\nH=\n{H}")
      print(selected_nodes)
      print(f"variable_degrees[{selected_nodes}]={variable_degrees[selected_nodes]}")
      #input("Press enter")

    # Set the corresponding entries in the parity-check matrix to 1
    H[i, selected_nodes] = 1
    variable_degrees[selected_nodes] += 1

  return H

import itertools
import random
'''
def permute_columns(matrix, p):
    # Get the number of columns in the matrix
    num_columns = len(matrix[0])
    
    # Generate all permutations of column indices
    column_permutations = itertools.permutations(range(num_columns))
    column_permutations_list=list(column_permutations)
    print(f"\ncolumn_permutations_list={column_permutations_list}")
    random.shuffle(column_permutations_list)
    print(f"\ncolumn_permutations_list after shuffle={column_permutations_list}")
    column_permutations=iter(column_permutations_list)
    #print(f"\ntype(column_permutations)={type(column_permutations)}")
    
    
    # Create a new matrix for the first p permutations of columns
    permuted_matrices = []
    for _ in range(p):
        print(f"\n_={_}")
        try:
            #perm = next(column_permutations)
            perm = next(column_permutations)
            print(f"perm={perm}")
            permuted_matrix = [[row[i] for i in perm] for row in matrix]
            permuted_matrices.append(permuted_matrix)
            print(f"permuted_matrices=\n{permuted_matrices}")
        except StopIteration as SI:
            print(f"\nException:{column_permutations}")
            break
        except Exception as e:
            print(f"\nException:{e}")
    
    return permuted_matrices
'''

def matrix_column_random_permutation_patterns(noOfColumns, noOfPatterns):
    
    # Generate all permutations of column indices
    column_permutations = itertools.permutations(range(noOfColumns))
    column_permutations_list=list(column_permutations)
    print("generating permutations")
    #print(f"\ncolumn_permutations_list={column_permutations_list}")
    random.shuffle(column_permutations_list)
    #print(f"\ncolumn_permutations_list after shuffle={column_permutations_list}")
    column_permutations=iter(column_permutations_list)
    #print(f"\ntype(column_permutations)={type(column_permutations)}")
    
    
    # Create a new matrix for the first p permutations of columns
    permutation_patterns=[]
    for _ in range(noOfPatterns):
        #print(f"\n_={_}")
        try:
            #perm = next(column_permutations)
            permutation_patterns.append(next(column_permutations))
            #print(f"permutation_patterns={permutation_patterns}")
        except StopIteration as SI:
            print(f"\nException:{column_permutations}")
            break
        except Exception as e:
            print(f"\nException:{e}")
    
    return permutation_patterns

def matrix_column_random_permutation_patterns_2(noOfColumns, noOfPatterns):
    
    # Create a new matrix for the first p permutations of columns
    #column_indexes=range()
    permutation_patterns=[]
    for _ in range(noOfPatterns):
        #print(f"\n_={_}")
        try:
            while(1):
                temp_pattern = list(range(noOfColumns))
                random.shuffle(temp_pattern)
                #print(f"temp_pattern={temp_pattern}")
                if temp_pattern not in permutation_patterns:
                    permutation_patterns.append(temp_pattern)
                    print(f"permutation_patterns={permutation_patterns}")
                    break
                
        except StopIteration as SI:
            print(f"\nException:{SI}")
            break
        except Exception as e:
            print(f"\nException:{e}")
    
    return permutation_patterns

def generate_gallager_ldpc_code2(n, k, d_c, d_r):
    """
    Generates a random Gallager LDPC code.

    Args:
    n: Code length.
    k: Number of information bits.
    d_v: Variable node degree.
    d_c: Check node degree.

    Returns:
    A numpy array representing the parity-check matrix H.
    """
    m = n-k
    H = np.zeros((m, n), dtype=int)



    H_0_rows = m/d_c
    print(f"\nH_0_rows={H_0_rows}")
    H_0_cols = n
    H_continuous_1s = int(n/d_r)
    print(f"\nH_continuous_1s={H_continuous_1s}")

    ##### Forming the H_0 part of H #####
    for i in range(1, int(H_0_rows+1)):
        #print(f"row={i}")
        for j in range((i-1)*d_r+1, i*d_r+1):
            #print(f"Column={j}")
            H[i-1,j-1] = 1

    print(f"\nH after H_0 is initialized\n{H}")
    

    noOfPatterns = d_c-1  # Number of permutations to generate. This is the number of sub matrices which are made out of H, except the H_0 matrix.
    permutation_patterns=matrix_column_random_permutation_patterns_2(len(H[0]), noOfPatterns)
    permuted_matrices=[]
    
    for index, patern in enumerate(permutation_patterns):
        permuted_matrix = [[row[i] for i in patern] for row in H[0:int(H_0_rows), :]]
        print(f"permuted_matrix=\n{permuted_matrix}")
        print(f"shape(H[int(H_0_rows*(index+1)):int(H_0_rows*(index+2)), :])={np.shape(H[int(H_0_rows*(index+1)):int(H_0_rows*(index+2)), :])}")
        print(f"np.shape(np.array(permuted_matrix))={np.shape(np.array(permuted_matrix))}")
        H[int(H_0_rows*(index+1)):int(H_0_rows*(index+2)), :] = np.array(permuted_matrix)
        permuted_matrices.append(permuted_matrix)

    #print(f"permuted_matrices=\n{np.array(permuted_matrices[0])}")

    return H

def formTannerGraph(H:np.ndarray):
    codeword_bits={}
    check_bits={}

    # Initialize codeword_bits
    for j in range(len(H[0])):
        codeword_bits[j]={'connectedTo':[], 'bit':-1, 'signalForChange':0, 'signalForNoChange':0}

    # Initialize check_bits
    for i in range(len(H[:,0:])):
        check_bits[i]={'connectedTo':[], 'codewordStream':[]}

    #print(f"codeword_bits=\n{codeword_bits}")
    #print(f"check_bits=\n{check_bits}")


    
    for i in range(len(H[:,0:])):
        for j in range(len(H[0])):
            if H[i,j]==1 :
                codeword_bits[j]['connectedTo'].append(i)
                check_bits[i]['connectedTo'].append(j)

    #print(f"codeword_bits=\n{codeword_bits}")

    return codeword_bits, check_bits

def isParityCheckOk(bits, parityType='even'):
    #paritySum=sum(bits)
    paritySum = cupy.sum(cupy.array(bits))
    isParityCheckOk=-1    # 1=Yes, 0=No
    if parityType=='even':
        if paritySum%2 == 0:
            isParityCheckOk = 1
        else:
            isParityCheckOk = 0
    elif parityType=='odd':
        if paritySum%2 == 1:
            isParityCheckOk = 1
        else:
            isParityCheckOk = 0
    return isParityCheckOk

def isIterationContinued(codeword_bits, check_bits, noOfIteration, noOfIteration_MAX):

    parityCheckCounter=0
    for checkNode in range(len(check_bits)):
        if isParityCheckOk([codeword_bits[codewordNode]['bit'] for codewordNode in check_bits[checkNode]['connectedTo']], 'even'):
            parityCheckCounter += 1
    '''
    if parityCheckCounter == len(check_bits):
        print(f"\nparityCheckCounter={parityCheckCounter}")
    if noOfIteration > noOfIteration_MAX:
        print(f"\nnoOfIteration={noOfIteration}")
    '''
    if parityCheckCounter == len(check_bits) or noOfIteration > noOfIteration_MAX:
        return 0
    else:
        return 1


def initializeCodewordNodes(codeword_bits, receivedVector):
    for index, bit in enumerate(receivedVector):
        codeword_bits[index]['bit']=bit

def calculateAndSendMessageToCheckNodes(codeword_bits, check_bits):

    for checkNode in range(len(check_bits)):
        check_bits[checkNode]['codewordStream'].clear()

    for codewordNode in range(len(codeword_bits)):
        for checkNode in codeword_bits[codewordNode]['connectedTo']:
            check_bits[checkNode]['codewordStream'].append(codeword_bits[codewordNode]['bit'])

    None

def calculateAndSendMessageToCodewordNodes(codeword_bits, check_bits, parityType='even'):
    
    for codewordNode in range(len(codeword_bits)):
        codeword_bits[codewordNode]['signalForChange'] = 0
        codeword_bits[codewordNode]['signalForNoChange'] = 0

    for checkNode in range(len(check_bits)):
        if isParityCheckOk(check_bits[checkNode]['codewordStream'], 'even')==1:
            hasTheCodewordBitToBeChanged = 0
        else:
            hasTheCodewordBitToBeChanged = 1

        '''
        paritySum=sum(check_bits[checkNode]['codewordStream'])
        hasTheCodewordBitToBeChanged=-1    # 1=Yes, 0=No
        if parityType=='even':
            if paritySum%2 == 0:
                hasTheCodewordBitToBeChanged = 0
            else:
                hasTheCodewordBitToBeChanged = 1
        elif parityType=='odd':
            if paritySum%2 == 1:
                hasTheCodewordBitToBeChanged = 0
            else:
                hasTheCodewordBitToBeChanged = 1
        '''
        for codewordNode in check_bits[checkNode]['connectedTo']:
            if hasTheCodewordBitToBeChanged==1:
                codeword_bits[codewordNode]['signalForChange'] += 1
            elif hasTheCodewordBitToBeChanged==0:
                codeword_bits[codewordNode]['signalForNoChange'] += 1
def codewordNodeBitFlip(codeword_bits):
    for codewordNode in range(len(codeword_bits)):
        if codeword_bits[codewordNode]['signalForChange'] > codeword_bits[codewordNode]['signalForNoChange'] :
            codeword_bits[codewordNode]['bit'] = codeword_bits[codewordNode]['bit'] ^ 1

def codewordNodeBitFlip2(codeword_bits): # incomplete
    codewordNodeToFlip = -1
    for codewordNode in range(len(codeword_bits)):
        if codeword_bits[codewordNode]['signalForChange'] > codeword_bits[codewordNode]['signalForNoChange']  and codeword_bits[codewordNode]['signalForChange'] > codeword_bits[codewordNodeToFlip]['signalForChange']:
            codewordNodeToFlip = codewordNode
    


def getDecodedCategory(originalCodeword, decoded_value, syndrome):
    category = 0
    originalCodeword=list(originalCodeword)
    decoded_value=list(decoded_value)
    #print(f"type(originalCodeword)={type(originalCodeword)}\ntype(decoded_value)={type(decoded_value)}")
    if originalCodeword == decoded_value:
        category = 1 # Decoded into same code-word
    #elif sum(syndrome) == 0:
    elif cupy.sum(cupy.array(syndrome)) == 0:
        category = 2 # Decoded into different but valid code-word
    else:
        category = 3 # Iterations are exausted but decoded into invalid code-word
    return category

def decodeWithHMP(codeword_bits, check_bits, receivedVector, noOfIteration, noOfIteration_MAX):
    initializeCodewordNodes(codeword_bits, receivedVector)
    #print(f"codeword_bits=\n{codeword_bits}")
    while isIterationContinued(codeword_bits, check_bits, noOfIteration, noOfIteration_MAX):
        calculateAndSendMessageToCheckNodes(codeword_bits, check_bits)
        calculateAndSendMessageToCodewordNodes(codeword_bits, check_bits, 'even')
        codewordNodeBitFlip(codeword_bits)
        noOfIteration+=1
def main_function():
    # Example usage:
    n = 20  # Code length
    k = 5   # Number of information bits
    m = n-k
    d_c = 3   # Variable node degree
    d_r = 4   # Check node degree

    '''
    # Example usage:
    n = 9  # Code length
    k = 3   # Number of information bits
    m = n-k
    d_c = 2   # Variable node degree
    d_r = 3   # Check node degree
    '''
    '''
    # Example usage:
    n = 10  # Code length
    m = 5
    k = n-m  # Number of information bits
    d_c = 2   # Variable node degree
    d_r = 4   # Check node degree
    '''
    #H = generate_gallager_ldpc_code2(n, k, d_c, d_r)
    H=np.array(
    [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]]
    )
    print(f"H=\n{H}")
    '''
    import ldpc.code_util
    #G = ldpc.code_util.construct_generator_matrix(H)
    #print(f"G=\n{G.toarray()}")
    G = get_generator_matrix(H)
    print(f"G=\n{G}")
    print(f"G@H.T=\n{(G@H.T)%2}")

    message=np.array([1,0,1])
    #codeword=message@
    '''
    ##### Now its time to form the tanner graph #####
    codeword_bits, check_bits = formTannerGraph(H)

    #receivedVector=np.array([0,1,1,1,0,0,0,1,1])
    #receivedVector=np.array([1,0,0,0,0])
    receivedVector=np.array([1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0])
    print(f"(receivedVector@H.T)%2=\n{(receivedVector@H.T)%2}")


    #####  #####
    noOfIteration_MAX = 10000
    noOfIteration = 1
    decodeWithHMP(codeword_bits, check_bits, receivedVector, noOfIteration, noOfIteration_MAX)

    print(f"\ncodeword_bits=\n{codeword_bits}")
    print(f"\ncheck_bits=\n{check_bits}")
    decoded_codeword=[codeword_bits[codewordNode]['bit'] for codewordNode in codeword_bits]
    print(f"\nDecoded Codeword={decoded_codeword}")
    print(f"(decode_codeword@H.T)%2=\n{(decoded_codeword@H.T)%2}")

    from ldpc import BpDecoder
    bpd = BpDecoder(
        H, #the parity check matrix
        error_rate=0.1, # the error rate on each bit
        max_iter=10000, #the maximum iteration depth for BP
        bp_method="product_sum", #BP method. The other option is `minimum_sum'
    )
    decoded_with_BP=bpd.decode(receivedVector)
    print(f"\nDecoded Codeword with BP={decoded_with_BP}")
    print(f"(decoded_with_BP@H.T)%2=\n{(decoded_with_BP@H.T)%2}")

    
'''
# Example parity-check matrix H (not in normal form)
import ldpc.code_util
H = np.array([
    [1, 0, 1, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 0, 0],
    [1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 1, 1]
])

H1 = np.array([ [1, 0, 1, 1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0, 0, 0], [0, 0, 1, 1, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1, 0, 1, 1], [1, 1, 1, 0, 0, 0, 1, 1, 0] ])

H2 = np.array(
            [
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 1]])

# Get the generator matrix G
#G1 = get_generator_matrix(H1)
#G2 = get_generator_matrix(H2)
G1 = ldpc.code_util.construct_generator_matrix(H1)
G2 = ldpc.code_util.construct_generator_matrix(H2)

print(f"\nParity-check matrix H1:\n{H1}")
print(f"\nGenerator matrix G1:\n{G1.toarray()}")
print(f"G1@H1.T=\n{(G1.toarray()@H1.T)%2}")

print(f"\nParity-check matrix H2:\n{H2}")
print(f"\nGenerator matrix G2:\n{G2.toarray()}")
print(f"G2@H2.T=\n{(G2.toarray()@H2.T)%2}")


H3=np.array(
[
[1, 0, 0, 0, 0,  0, 0, 0, 0, 0,  1, 0, 1, 0, 0,  0, 1, 0, 1, 1],
[0, 1, 0, 0, 0,  0, 0, 0, 0, 0,  0, 1, 0, 1, 0,  1, 0, 1, 0, 1],
[0, 0, 1, 0, 0,  0, 0, 0, 0, 0,  1, 0, 0, 1, 0,  0, 1, 1, 0, 1],
[0, 0, 0, 1, 0,  0, 0, 0, 0, 0,  0, 0, 1, 0, 1,  1, 1, 0, 1, 0],
[0, 0, 0, 0, 1,  0, 0, 0, 0, 0,  0, 1, 0, 0, 1,  1, 0, 1, 1, 0],

[1, 0, 1, 0, 0,  0, 1, 0, 1, 1,  1, 0, 0, 0, 0,  0, 0, 0, 0, 0],
[0, 1, 0, 1, 0,  1, 0, 1, 0, 1,  0, 1, 0, 0, 0,  0, 0, 0, 0, 0],
[1, 0, 0, 1, 0,  0, 1, 1, 0, 1,  0, 0, 1, 0, 0,  0, 0, 0, 0, 0],
[0, 0, 1, 0, 1,  1, 1, 0, 1, 0,  0, 0, 0, 1, 0,  0, 0, 0, 0, 0],
[0, 1, 0, 0, 1,  1, 0, 1, 1, 0,  0, 0, 0, 0, 1,  0, 0, 0, 0, 0]
]
)
G3=ldpc.code_util.construct_generator_matrix(H3)

print(f"\nParity-check matrix H3:\n{H3}")
print(f"G3=\n{G3.toarray()}")
print(f"G3@H3.T=\n{(G3.toarray()@H3.T)%2}")

## find out it is why that for same shape, different shape G matrices are produced
'''

#main_function()