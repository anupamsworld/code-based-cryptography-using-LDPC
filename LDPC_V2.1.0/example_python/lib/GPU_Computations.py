def GF2MatrixMultUsingGpu(A, B):
    import cupy
    A_gpu = cupy.asarray(A)
    B_gpu = cupy.asarray(B)
    AB_gpu = cupy.matmul(A_gpu, B_gpu)
    AB_gpu = AB_gpu % 2
    return cupy.asnumpy(AB_gpu)

def GF2MatrixAddUsingGPU(A, B):
    import cupy
    A_gpu = cupy.asarray(A)
    B_gpu = cupy.asarray(B)
    '''
    AplusB = cupy.add(A_gpu, B_gpu)
    AplusB = AplusB % 2
    '''
    AplusB = cupy.where(A_gpu == B_gpu, 0, 1)
    return cupy.asnumpy(AplusB)

def GF2makeErrorMatrixUsingGPU(n, noOfErrorBits=None, method=1, bitErrorProb=0.1, errorIndexesSet=None):
    import random
    import cupy, numpy
    if noOfErrorBits is None:
        noOfErrorBits = n * bitErrorProb
    if method==1 :
        # Generate random floats between 0 and 1
        #random_floats = cupy.asarray(numpy.full(n, bitErrorProb))
        random_floats = cupy.random.rand(n)

        # Convert floats to binary values (0 or 1)
        binary_array = cupy.where(random_floats > (noOfErrorBits/n), 0, 1)
        #binary_array = cupy.where(random_floats > 0.01223091976516634050880626223092, 0, 1) #as 100/8176=0.01223091976516634050880626223092
        #binary_array = cupy.where(random_floats > 0.00110078277886497064579256360078, 0, 1) #as 9/8176=0.00110078277886497064579256360078
        #binary_array = cupy.where(random_floats > 0.00122309197651663405088062622309, 0, 1) #as 10/8176=0.00122309197651663405088062622309
        #binary_array = cupy.where(random_floats > 0.00244618395303326810176125244618, 0, 1) #as 20/8176=0.00244618395303326810176125244618
        #binary_array = cupy.where(random_floats > 0.01002935420743639921722113502935, 0, 1) #as 82/8176=0.01002935420743639921722113502935
        #binary_array = cupy.where(random_floats > 0.01015166340508806262230919765166, 0, 1) #as 83/8176=0.01015166340508806262230919765166
        #binary_array = cupy.where(random_floats > 0.00917318982387475538160469667319, 0, 1) #as 75/8176=0.00917318982387475538160469667319
        #binary_array = cupy.where(random_floats > 0.00905088062622309197651663405088, 0, 1) #as 74/8176=0.00905088062622309197651663405088
        #binary_array = cupy.where(random_floats > 0.00892857142857142857142857142857, 0, 1) #as 73/8176=0.00892857142857142857142857142857
        #binary_array = cupy.where(random_floats > 0.00880626223091976516634050880626, 0, 1) #as 72/8176=0.00880626223091976516634050880626
        #binary_array = cupy.where(random_floats > 0.00868395303326810176125244618395, 0, 1) #as 71/8176=0.00868395303326810176125244618395
        #binary_array = cupy.where(random_floats > 0.00856164383561643835616438356164, 0, 1) #as 70/8176=0.00856164383561643835616438356164
        #binary_array = cupy.where(random_floats > 0.00843933463796477495107632093933, 0, 1) #as 69/8176=0.00843933463796477495107632093933
        #binary_array = cupy.where(random_floats > 0.00831702544031311154598825831703, 0, 1) #as 68/8176=0.00831702544031311154598825831703
        #binary_array = cupy.where(random_floats > 0.00819471624266144814090019569472, 0, 1) #as 67/8176=0.00819471624266144814090019569472
        #binary_array = cupy.where(random_floats > 0.00807240704500978473581213307241, 0, 1) #as 66/8176=0.00807240704500978473581213307241
        #binary_array = cupy.where(random_floats > 0.0079500978473581213307240704501, 0, 1) #as 65/8176=0.0079500978473581213307240704501
        #binary_array = cupy.where(random_floats > 0.00978473581213307240704500978474, 0, 1) #as 80/8176=0.00978473581213307240704500978474
        #binary_array = cupy.where(random_floats > 0.01125244618395303326810176125245, 0, 1) #as 92/8176=0.01125244618395303326810176125245
        #binary_array = cupy.where(random_floats > 0.000611545988258317, 0, 1) #as 5/8176=0.000611545988258317
        #binary_array = cupy.where(random_floats > 0.0092954990215264187866927592955, 0, 1) #as 76/8176=0.0092954990215264187866927592955
        #binary_array = cupy.where(random_floats > 0.00941780821917808219178082191781, 0, 1) #as 77/8176=0.00941780821917808219178082191781
        #binary_array = cupy.where(random_floats > 0.00990704500978473581213307240705, 0, 1) #as 81/8176=0.00990704500978473581213307240705
        #binary_array = cupy.where(random_floats > 0.00966242661448140900195694716243, 0, 1) #as 79/8176=0.00966242661448140900195694716243
        #binary_array = cupy.where(random_floats > 0.00954011741682974559686888454012, 0, 1) #as 78/8176=0.00954011741682974559686888454012
        #binary_array = cupy.where(random_floats > 0.01223091976516634050880626223092, 0, 1) #as 100/8176=0.01223091976516634050880626223092
        #binary_array = cupy.where(random_floats > 0.01100782778864970645792563600783, 0, 1) #as 90/8176=0.01100782778864970645792563600783
        #binary_array = cupy.where(random_floats > 0.48923679060665362035225048923679, 0, 1) #as 4000/8176=0.48923679060665362035225048923679
        errorIndexes = cupy.where(binary_array==1)

        #print(f"type(indexes)= {type(indexes)}")

        return cupy.asnumpy(binary_array), errorIndexes[0]
    elif method==2:
        errorIndexes=[]
        binary_array=[0]*n
        for index in range(n):
            if index % 1200 == 0 :
                binary_array[index] = 1
                errorIndexes.append(index)
        return cupy.asarray(binary_array), cupy.asarray(errorIndexes)
    elif method==3:
        """
        Extracts p random elements from a set, returns the remaining set and a list of selected elements.

        Parameters:
        s (set): The input set.
        p (int): Number of random elements to extract.

        Returns:
        tuple: A tuple containing the modified set and the list of extracted elements.
        """
        errorIndexes=[]
        if errorIndexesSet is None or (len(errorIndexesSet) > n and len(errorIndexesSet) <= 0):
            return cupy.asnumpy([0]*n), errorIndexesSet, cupy.asnumpy([])
        if noOfErrorBits > len(errorIndexesSet):
            print("noOfErrorBits cannot be greater than the size of the set")
            return cupy.asnumpy([0]*n), errorIndexesSet, cupy.asnumpy([])

        selected_elements = random.sample(sorted(errorIndexesSet), noOfErrorBits)
        errorIndexesSet -= set(selected_elements)

        # Use CuPy for GPU processing
        error = cupy.zeros(n, dtype=cupy.int32)
        selected_elements_gpu = cupy.array(selected_elements)
        error[selected_elements_gpu] = 1

        errorIndexes = cupy.where(error==1)

        return cupy.asnumpy(error), errorIndexesSet, errorIndexes[0]


'''
#import numpy
#print(f"{GF2MatrixAddUsingGPU(numpy.array([1,0]), numpy.array([1,1]))}")
'''