




def  fetch_matrix_from_file(fullFilePath, fetchType="allAtOnce"):
    import numpy as np
    '''
    import sys,os
    PWD=os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.abspath(PWD+'/'))
    '''
    from . import util

    list = []
    if fetchType == "rowAtOnce":
      with open(fullFilePath, "r") as file:
            for line in file:
                # Process each line here
                #print(line.strip())  # Remove leading/trailing whitespace
                list.append(util.binary_string_to_int_list_no_spaces(line.strip()))
    elif fetchType == "allAtOnce":
      with open(fullFilePath, "r") as file:
        file_content = file.read()
      
      # Split the string into a list of lines
      lines = file_content.splitlines()

      # Iterate over each line
      for line in lines:
          # Process each line here
          list.append(util.binary_string_to_int_list_no_spaces(line.strip()))

    matrix = np.array(list)

    #print(f"Formed matrix has shape={matrix.shape}")

    return matrix



def write_matrix_to_file(array_or_list, filename, dimention):
    import os
    PWD=os.path.dirname(os.path.realpath(__file__))

    if dimention==2:
        with open(PWD+'/'+filename+'.txt', 'w') as file:
            for i in range(array_or_list.shape[0]):
                for j in range(array_or_list.shape[1]):
                    file.write(str(array_or_list[i][j]))
                file.write("\n")
    elif dimention==1:
        with open(PWD+'/'+filename+'.txt', 'w') as file:
            for i in range(array_or_list.shape[0]):
                file.write(str(array_or_list[i]))

def write_str_to_file(str, filename):
    import os
    PWD=os.path.dirname(os.path.realpath(__file__))

    with open(PWD+'/'+filename+'.txt', 'w') as file:
            file.write(str)


def read_array_from_file(file_path, element_type="string"):
    import numpy as np
    with open(file_path, 'r') as file:
        content = file.read()
        content = content.strip("[")
        content = content.strip("]")
        if(element_type == "int"):
            items = [int(item) for item in content.split()]
        else:
            items = content.split()  # split by whitespace (space, newline, tab)
        #return items
        if element_type == "int":
            return np.array(items, dtype=int)
        else:
            return np.array(items, dtype=str)

def binaryArrayFromIndices(indices: str, length:int, element_type="int"):
    """
    Creates a binary array from indices.
    
    Returns:
        A binary array with 1s at specified indices and 0s elsewhere.
    """
    import numpy as np
    indices = indices.strip("[")
    indices = indices.strip("]")
    if(element_type == "int"):
        items = [int(item) for item in indices.split()]
    else:
        items = indices.split()  # split by whitespace (space, newline, tab)
    binary_array = np.zeros(length+1, dtype=int)
    binary_array[items] = 1
    return binary_array