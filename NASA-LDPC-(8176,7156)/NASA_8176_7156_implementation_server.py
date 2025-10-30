# LDPC package Version: 2.2.8
print("\nInitializing Communication System Server, powered by NASA LDPC (8176, 7156) Code based McEliece encryption...")

import json
import parity_check_matrix as PCM
import generator_matrix as GM
import numpy as np
import os
import sys
PWD=os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(PWD+'/../'))
import cupy
import lib.GPU_Computations as GPUC
import lib.util as utl
import Bit_Flipping.Hard_Message_Passing as HMP
import lib.store_and_fetch as sf
import hashlib

from flask import Flask, request
app = Flask(__name__)
#import logging
#log = logging.getLogger('werkzeug')
#log.setLevel(logging.ERROR)  # or logging.CRITICAL

# Example usage:
n = 8176  # Code length
k = 7154   # Number of information bits
m = n-k

#noOfErrorBits=90
noOfIteration_MAX = 10000
error_rate = 80/8176
#exit()

H = np.array(PCM.H_list)
G = np.array(GM.G_list)

userInputMessageSegment = ""  # This will be set from the request data
cumulativeTextOfUserInformation_atReceiver = ""
P_inverse = sf.fetch_matrix_from_file(PWD+"/P_inverse_matrix.txt")
S_inverse = sf.fetch_matrix_from_file(PWD+"/S_inverse_matrix.txt")

def processAtReceiver():
    ##### Start of Procedure at Receiver #####

    
    ##### Start of Decryption at Receiver #####
    global userInputMessageSegment
    global H,G, P_inverse, S_inverse, n, k, noOfIteration_MAX, error_rate
    y_atReceiver_matrix = np.array(utl.binary_string_to_int_list_no_spaces(userInputMessageSegment)).astype(int)  # Convert the user input message segment to a numpy array of integers
    #print(f"\nshape of y_atReceiver_matrix={y_atReceiver_matrix.shape}")
    #print(f"shape of P_inverse={P_inverse.shape}")
    yp_inverse_atReceiver_matrix = GPUC.GF2MatrixMultUsingGpu(y_atReceiver_matrix, P_inverse)
    from ldpc import BpDecoder
    bpd = BpDecoder(
        #bpd = bp_decoder(
            H, #the parity check matrix
            error_rate=float(error_rate), # the error rate on each bit
            max_iter=noOfIteration_MAX, #the maximum iteration depth for BP
            input_vector_type="received_vector"
        )
    messageSG_decoded_atReceiver_matrix = bpd.decode(np.array(yp_inverse_atReceiver_matrix)) # mS is the BP_decoded_codeword
    #print(f"messageSG_decoded_atReceiver_matrix category={HMP.getDecodedCategory(messageSG_atSender_matrix, messageSG_decoded_atReceiver_matrix, GPUC.GF2MatrixMultUsingGpu(messageSG_decoded_atReceiver_matrix, H.T))}")

    decoded_messageS_atReceiver_matrix = None
    isReTransmissionNeeded = False
    responseMessage = ""
    #if sum((messageSG_decoded_atReceiver_matrix@H.T)%2) == 0:
    syndromSum = cupy.sum(GPUC.GF2MatrixMultUsingGpu(messageSG_decoded_atReceiver_matrix, H.T))
    #print(f"syndromSum={syndromSum}")
    if syndromSum == 0:
        #print(f"\n{messageSG_decoded_atReceiver_matrix} is a valid deoded codeword")
        #print(f"\nDecoded into a valid codeword")
        decoded_messageS_atReceiver_matrix = utl.getMessageFromSystematicCodeword(n, k, messageSG_decoded_atReceiver_matrix)
        #print(f"{len(decoded_messageS_atReceiver_matrix)}")
        #print(f"Decrypted original messageS={decoded_messageS_atReceiver_matrix}")

        #decrypted_msg = ((np.array(list(decoded_messageS_atReceiver_matrix)).astype(int)) @ S_inverse) % 2
        decrypted_message_atReceiver_matrix = GPUC.GF2MatrixMultUsingGpu(np.array(list(decoded_messageS_atReceiver_matrix)).astype(int), S_inverse)
        #print(f"The decryped message={decrypted_message_atReceiver_matrix}")

        # Now Check for hash

        binaryOfMessage_atReceiver = ''.join([str(bit) for bit in decrypted_message_atReceiver_matrix])
        global cumulativeTextOfUserInformation_atReceiver
        textOfUserInformation_atReceiver = utl.binary_to_text(binaryOfMessage_atReceiver[2:6898]).strip('\x00')
        #print("\nCalculated information of this segment is:\n"+textOfUserInformation_atReceiver)
        cumulativeTextOfUserInformation_atReceiver += textOfUserInformation_atReceiver
        
        #print("\nlength of textOfUserInformation_atReceiver= "+str(len(textOfUserInformation_atReceiver)))
        hexHashofInformation_atReceiver = utl.binary_to_hex(binaryOfMessage_atReceiver[6898:])
        #print("\n"+hexHashofInformation_atReceiver)
        print("\n")
        ##### Check Hash Start #####
        h_r = hashlib.new("sha256")
        h_r.update(textOfUserInformation_atReceiver.encode('utf-8'))
        hexHashofInformationComputed_atReceiver = h_r.hexdigest()
        #print("Hash of information at sender: ", hexHashOfInformation_atSender)
        #print("Hash of information at receiver: ", hexHashofInformation_atReceiver)
        #print("Hash of information at receiver (computed): ", hexHashofInformationComputed_atReceiver)
        if hexHashofInformationComputed_atReceiver == hexHashofInformation_atReceiver:
            print("Hash is verified at the receiver.")
            isReTransmissionNeeded = False
        else:
            responseMessage = "Hash is not verified at the receiver."
            print(responseMessage)
            isReTransmissionNeeded = True
    else:
        #print(f"\n{messageSG_decoded_atReceiver_matrix} is not a valid deoded codeword")
        responseMessage = "Decoded into a invalid codeword."
        print(responseMessage)
        isReTransmissionNeeded = True
    print("\nCumulative information at receiver:\n<<< "+cumulativeTextOfUserInformation_atReceiver+" >>>")
    return isReTransmissionNeeded, responseMessage
    ##### End of Decryption at Receiver #####

    

    ##### End of Procedure at Receiver #####

@app.route("/")
def hello_world():
    print("Hello, World!")
    return "<p>Hello, World!</p>"

@app.route("/nasa-8176-7156-communication/", methods=['POST'])
def nasa_8176_7156_communication():
    if request.method == 'POST':
        requestDataDictionary = json.loads(request.data.strip())
        #print("\nReceived data:", requestDataDictionary)
        #print("\nType of requestDataDictionary:", type(requestDataDictionary))
        #print("\nType of requestDataDictionary[\"msg_seg_num\"]:", type(requestDataDictionary["msg_seg_num"]))

        messageSegmentNumber = int(requestDataDictionary["msg_seg_num"])
        totalMessageSegments = int(requestDataDictionary["total_msg_seg"])
        global cumulativeTextOfUserInformation_atReceiver
        if requestDataDictionary["command"] == "start_com" :
            cumulativeTextOfUserInformation_atReceiver = ""  # Reset the text of information at receiver
            print("\nReceiving new message.")
        #for i in range(totalMessageSegments):
        print("\nProcessing message segment number:", messageSegmentNumber)
        global userInputMessageSegment
        userInputMessageSegment = requestDataDictionary["msg_seg"]
        if len(userInputMessageSegment) != 8176:
            print("Error: Length of message segment is not 8176 bits. It is:", len(userInputMessageSegment))
            return "<p>Error: Length of message segment is not 8176 bits.</p>"
        isTransmissionNeeded, responseMessage = processAtReceiver()
        if isTransmissionNeeded:
            print(f"Re-transmission is needed for segment number:{messageSegmentNumber}\n")
            # Here we can send a message to the sender to re-transmit the message
            return "{\"ack\":\""+ str(messageSegmentNumber) +"\", \"responseMessage\":\"" + responseMessage + "\"}"
        else:
            print(f"Re-transmission is NOT needed for segment number:{messageSegmentNumber}\n")
            # Here we can send a message to the sender that no re-transmission is needed
            return "{\"ack\":\""+ str(messageSegmentNumber+1) +"\", \"responseMessage\":\"" + responseMessage + "\"}"



    else:
        print("Method not allowed. Please use POST method.")
        return "<p>Method not allowed. Please use POST method.</p>"
    print("\nnasa-8176-7156-communication.")
    return "<p>nasa-8176-7156-communication.</p>"

'''
start_time = time.time()

# Example usage:
n = 8176  # Code length
k = 7154   # Number of information bits
m = n-k

noOfErrorBits=90
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
    print("\n"+binaryOfInformation_atSender)
    print("\n"+textOfUserInformation_atSender)

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
    codeword_atSender = (binaryOfMessage_atSender_matix@G)%2

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
            bp_method="product_sum", #BP method. The other option is `minimum_sum'
            schedule = "serial" # the BP schedule
        )
    messageSG_decoded_atReceiver_matrix = bpd.decode(np.array(yp_inverse_atReceiver_matrix)) # mS is the BP_decoded_codeword
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
'''