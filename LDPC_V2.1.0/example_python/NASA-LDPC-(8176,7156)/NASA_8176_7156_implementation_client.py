# LDPC package Version: 2.2.8
print("\nInitializing Communication System Client, powered by NASA LDPC (8176, 7156) Code based McEliece encryption...")

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
import hashlib
import requests


url = "http://127.0.0.1:5000/nasa-8176-7156-communication/"

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



start_time = time.time()

# Example usage:
n = 8176  # Code length
k = 7154   # Number of information bits
m = n-k

noOfErrorBits=80
noOfIteration_MAX = 10000

#exit()

H = np.array(PCM.H_list)
G = np.array(GM.G_list)

##### Start of Procedure at Sender #####


##### Start of matrices initializations for Sender #####

S = sf.fetch_matrix_from_file(PWD+"/S_matrix.txt")
P = sf.fetch_matrix_from_file(PWD+"/P_matrix.txt")


##### End of matrices initializations for Sender #####

while True:
    userInputMessage = input("\nEnter the message to be sent : ")
    print("\nLength of user input message: ", len(userInputMessage))
    # chop the whole text into multiple segments of 862 characters each
    messageSegments = [userInputMessage[i:i+862] for i in range(0, len(userInputMessage), 862)]

    for messageSegmentIndex, messageSegmentText in enumerate(messageSegments):

        #text = "Hi there. How are you?"
        print("\nProcessing message segment number:", messageSegmentIndex + 1)
        print(f"\nlength of segment {messageSegmentIndex + 1}= "+str(len(messageSegmentText)))
        binaryOfInformation_atSender = utl.text_to_binary(messageSegmentText)
        binaryOfInformation_atSender = binaryOfInformation_atSender.zfill(6898)
        textOfUserInformation_atSender = utl.binary_to_text(binaryOfInformation_atSender[2:])
        #print("\n"+binaryOfInformation_atSender)
        #print("\n"+textOfUserInformation_atSender)

        ##### Check Complete #####

        h = hashlib.new("sha256")
        h.update(messageSegmentText.encode('utf-8'))
        hexHashOfInformation_atSender = h.hexdigest()
        #print("\nHash of information at sender: ", hexHashOfInformation_atSender)
        #print("\n2nd time h.hexdigest(): "+h.hexdigest())
        binaryHashOfInformation_atSender = utl.hex_to_binary(hexHashOfInformation_atSender)
        #print("\nlength of binaryHashOfInformation_atSender: ", len(binaryHashOfInformation_atSender))
        binaryOfMessage_atSender = binaryOfInformation_atSender + binaryHashOfInformation_atSender
        #print("\n"+binaryOfMessage_atSender)


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
        while(True):
            e, error_positions = GPUC.GF2makeErrorMatrixUsingGPU(n, noOfErrorBits=noOfErrorBits, method=1)
            sum_e = cupy.sum(e)
            error_rate = sum_e/n
            y_atSender_matrix = GPUC.GF2MatrixAddUsingGPU(messageG_star_atSender_matrix, e)
            data={"command": "start_com" if messageSegmentIndex == 0 else "continue_com",
                "msg_seg_num": messageSegmentIndex+1,
                "total_msg_seg": len(messageSegments),
                "error_bits": "80",
                "msg_seg":"".join([str(bit) for bit in y_atSender_matrix])
                }
            response = requests.post(url, json=data)
            print("Status Code:", response.status_code)
            print("Acknowledgment Number:", response.json().get("ack"))
            #print("Response Body:", response.text)
            if response.status_code == 200:
                print("Message segment is sent successfully.")
                response_data = response.json()
                if int(response_data.get("ack")) == messageSegmentIndex+2:
                    break
                elif int(response_data.get("ack")) == messageSegmentIndex+1:
                    print("Receiver: "+ response_data.get("responseMessage"))
            else:
                print("Error sending message, retrying...\nPress Ctrl+C to stop.")
                time.sleep(1)


        ##### End of Encryption at Sender #####


        ##### End of Procedure at Sender #####

    print("\nMessage is sent successfully.")
    print("\nDo you want to send another message? (Y/N)")
    continueSending = input().strip().lower()
    if continueSending == 'n':
        print("Exiting the sender program.")
        break
    elif continueSending == 'y':
        print("\nContinuing to send another message...")


