import numpy as np
import copy
def generate_binary_strings(n):
    """Generates all 2^n binary strings of length n.

    Args:
    n: The length of the binary strings to generate.

    Returns:
    A list of all 2^n binary strings of length n.
    """

    if n == 0:
        return [""]

    binary_strings = []
    for string in generate_binary_strings(n - 1):
        binary_strings.append(string + "0")
        binary_strings.append(string + "1")

    return binary_strings


def generate_all_mS(all_msg, S):
    all_msgS = []
    for msg in all_msg:
        codeword = (np.array(list(msg), dtype=int) @ S) % 2
        all_msgS.append(''.join(map(str, codeword)))
    return all_msgS

def makeMessageCodewordTable(k, G):
    all_msg = generate_binary_strings(k)
    #print(f"all_msg=\n{all_msg}")
    messageCodewordTable = {}
    for msg in all_msg:
        codeword = (np.array(list(copy.deepcopy(msg)), dtype=int) @ G) % 2
        messageCodewordTable[''.join(map(str, list(codeword)))] = copy.deepcopy(msg)
    return messageCodewordTable

S = np.array(
[[1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
 [0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
 [1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
 [0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
 [1, 0, 0, 0, 1, 1, 0, 0, 1, 0],
 [0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
 [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
 [0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
 [1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
 [1, 0, 0, 1, 1, 1, 0, 1, 0, 0]]
)
# this is invertible

print(f"S=\n{S}")

msg = np.array([1, 0, 1, 1, 0, 0, 0, 0, 1, 0])
print(f"msg=\n{msg}")

msg2 = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1, 0])
print(f"msg2=\n{msg2}")

all_msg = generate_binary_strings(len(msg))
print(f"type(all_msg)={type(all_msg)}")
#print(f"all_msg=\n{all_msg}")
print(f"Total count of all_msg={len(all_msg)}")
print(f"Total unique count of all_msg={len(set(all_msg))}")

all_msgS = generate_all_mS(all_msg, S)
print(f"type(all_msgS)={type(all_msgS)}")
print(f"Total count of all_msgS={len(all_msgS)}")
print(f"Total unique count all_msgS={len(set(all_msgS))}")


G = np.array(
[[1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],       
 [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],       
 [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],       
 [0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],       
 [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],       
 [0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],       
 [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],       
 [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],       
 [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],       
 [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
)

messageCodewordTable = makeMessageCodewordTable(len(msg), G)

print(f"len(messageCodewordTable)={len(messageCodewordTable)}")
#print(f"messageCodewordTable=\n{messageCodewordTable}")

msgG = (msg@G)%2
print(f"msgG=\n{msgG}")

msg2G = (msg2@G)%2
print(f"msg2G=\n{msg2G}")

found_msg_from_table = messageCodewordTable[''.join(map(str, list(msgG)))]
print(f"for codeword {''.join(map(str, list(msgG)))} found_msg_from_table=\n{found_msg_from_table}")