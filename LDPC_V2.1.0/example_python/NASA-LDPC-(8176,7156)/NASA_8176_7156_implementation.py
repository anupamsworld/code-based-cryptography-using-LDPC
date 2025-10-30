# LDPC package Version: 2.2.8

import parity_check_matrix as PCM
import generator_matrix as GM
import numpy as np
import os
import sys
PWD=os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(PWD+'/../'))
import time
import cupy
import lib.GPU_Computations as GPUC
import lib.util as utl
import Bit_Flipping.Hard_Message_Passing as HMP
import lib.store_and_fetch as sf
import ldpc.mod2
import ldpc.code_util
import hashlib

start_time = time.time()

# Example usage:
n = 8176  # Code length
k = 7154   # Number of information bits
m = n-k

noOfErrorBits=92
noOfIteration_MAX = 10000

#exit()

H = np.array(PCM.H_list)
G = np.array(GM.G_list)

##### Start of Procedure at Sender #####
##### Start of Check #####
userInputMessage = input("Enter the message to be sent : ")
# chop the whole text into multiple segments of 862 characters each
messageSegments = [userInputMessage[i:i+862] for i in range(0, len(userInputMessage), 862)]


##### Start of matrices initializations for Sender #####

S = sf.fetch_matrix_from_file(PWD+"/S_matrix.txt")
P = sf.fetch_matrix_from_file(PWD+"/P_matrix.txt")
P_inverse = sf.fetch_matrix_from_file(PWD+"/P_inverse_matrix.txt")

##### End of matrices initializations for Sender #####

##### Start of matrices initializations for Receiver #####

S_inverse = sf.fetch_matrix_from_file(PWD+"/S_inverse_matrix.txt")

##### End of matrices initializations for Receiver #####


for text in messageSegments:

    #text = "Hi there. How are you?"
    print("\nlength of text= "+str(len(text)))
    binaryOfInformation_atSender = utl.text_to_binary(text)
    binaryOfInformation_atSender = binaryOfInformation_atSender.zfill(6898)
    textOfUserInformation_atSender = utl.binary_to_text(binaryOfInformation_atSender[2:])
    print("\nbinaryOfInformation_atSender: "+binaryOfInformation_atSender)
    print("\ntextOfUserInformation_atSender: "+textOfUserInformation_atSender)

    ##### Check Complete #####

    h = hashlib.new("sha256")
    h.update(text.encode('utf-8'))
    hexHashOfInformation_atSender = h.hexdigest()
    print("\nHash of information at sender: ", hexHashOfInformation_atSender)
    print("\n2nd time h.hexdigest(): "+h.hexdigest())
    binaryHashOfInformation_atSender = utl.hex_to_binary(hexHashOfInformation_atSender)
    print("\nlength of binaryHashOfInformation_atSender: ", len(binaryHashOfInformation_atSender))
    binaryOfMessage_atSender = binaryOfInformation_atSender + binaryHashOfInformation_atSender
    print("\n"+binaryOfMessage_atSender)


    ##### Start of Code-Word generation at Sender #####
    binaryOfMessage_atSender_matix = utl.binary_string_to_int_list_no_spaces(binaryOfMessage_atSender)
    codeword_atSender_matrix = (binaryOfMessage_atSender_matix@G)%2
    #print(f"Codeword at Sender={''.join([str(bit) for bit in codeword_atSender_matrix])}")
    

    ##### End of Code-Word generation at Sender #####

    ##### Start of Encryption at Sender #####

    
    
    messageS_atSender_matrix = GPUC.GF2MatrixMultUsingGpu(binaryOfMessage_atSender_matix, S)
    messageSG_atSender_matrix = GPUC.GF2MatrixMultUsingGpu(messageS_atSender_matrix, G)
    messageG_atSender_matrix = GPUC.GF2MatrixMultUsingGpu(binaryOfMessage_atSender_matix, G)
    SG_atSender_matrix = GPUC.GF2MatrixMultUsingGpu(S, G)
    G_star_atSender_matrix = GPUC.GF2MatrixMultUsingGpu(SG_atSender_matrix, P)
    messageG_star_atSender_matrix = GPUC.GF2MatrixMultUsingGpu(binaryOfMessage_atSender_matix, G_star_atSender_matrix)
    e, error_positions = GPUC.GF2makeErrorMatrixUsingGPU(n, method=1)
    sum_e = cupy.sum(e)
    error_rate = sum_e/n
    y_atSender_matrix = GPUC.GF2MatrixAddUsingGPU(messageG_star_atSender_matrix, e)
    print(f"\ny_atSender_matrix={''.join([str(bit) for bit in y_atSender_matrix])}")



    ##### End of Encryption at Sender #####


    ##### End of Procedure at Sender #####







    ##### Start of Procedure at Receiver #####

    
    ##### Start of Decryption at Receiver #####

    yp_inverse_atReceiver_matrix = GPUC.GF2MatrixMultUsingGpu(y_atSender_matrix, P_inverse)
    from ldpc import BpDecoder
    bpd = BpDecoder(
        #bpd = bp_decoder(
            H, #the parity check matrix
            error_rate=float(error_rate), # the error rate on each bit
            max_iter=noOfIteration_MAX, #the maximum iteration depth for BP
            input_vector_type="received_vector"
        )
    messageSG_decoded_atReceiver_matrix = bpd.decode(np.array(yp_inverse_atReceiver_matrix)) # mSG is the BP_decoded_codeword
    print(f"messageSG_decoded category={HMP.getDecodedCategory(messageSG_atSender_matrix, messageSG_decoded_atReceiver_matrix, GPUC.GF2MatrixMultUsingGpu(messageSG_decoded_atReceiver_matrix, H.T))}")

    decoded_messageS_atReceiver_matrix = None
    #if sum((messageSG_decoded@H.T)%2) == 0:
    if sum(GPUC.GF2MatrixMultUsingGpu(messageSG_decoded_atReceiver_matrix, H.T)) == 0:
        print(f"\n{messageSG_decoded_atReceiver_matrix} is a valid deoded codeword")
        decoded_messageS_atReceiver_matrix = utl.getMessageFromSystematicCodeword(n, k, messageSG_decoded_atReceiver_matrix)
        print(f"{len(decoded_messageS_atReceiver_matrix)}")
        print(f"Decrypted original messageS={decoded_messageS_atReceiver_matrix}")

        #decrypted_msg = ((np.array(list(decoded_messageS_atReceiver_matrix)).astype(int)) @ S_inverse) % 2
        decrypted_message_atReceiver_matrix = GPUC.GF2MatrixMultUsingGpu(np.array(list(decoded_messageS_atReceiver_matrix)).astype(int), S_inverse)
        print(f"The decryped message={decrypted_message_atReceiver_matrix}")


    else:
        print(f"\n{messageSG_decoded_atReceiver_matrix} is not a valid deoded codeword")

    ##### End of Decryption at Receiver #####

    binaryOfMessage_atReceiver = ''.join([str(bit) for bit in decrypted_message_atReceiver_matrix])
    textOfUserInformation_atReceiver = utl.binary_to_text(binaryOfMessage_atReceiver[2:6898]).strip('\x00')
    print("\n"+textOfUserInformation_atReceiver)
    print("\nlength of textOfUserInformation_atReceiver= "+str(len(textOfUserInformation_atReceiver)))
    hexHashofInformation_atReceiver = utl.binary_to_hex(binaryOfMessage_atReceiver[6898:])
    print("\n"+hexHashofInformation_atReceiver)
    print("\n")
    ##### Check Hash Start #####
    h_r = hashlib.new("sha256")
    h_r.update(textOfUserInformation_atReceiver.encode('utf-8'))
    hexHashofInformationComputed_atReceiver = h_r.hexdigest()
    print("Hash of information at sender: ", hexHashOfInformation_atSender)
    print("Hash of information at receiver: ", hexHashofInformation_atReceiver)
    print("Hash of information at receiver (computed): ", hexHashofInformationComputed_atReceiver)
    if hexHashofInformationComputed_atReceiver == hexHashofInformation_atReceiver:
        print("Hash is verified at the receiver")
    else:
        print("Hash is not verified at the receiver")

    ##### End of Procedure at Receiver #####
