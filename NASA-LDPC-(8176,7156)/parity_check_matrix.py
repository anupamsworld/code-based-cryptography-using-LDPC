import numpy as np
import os
circulants =[
    [
        [0,176],
        [523,750],
        [1022,1374],
        [1557,1964],
        [2044,2436],
        [2706,2964],
        [3066,3417],
        [3586,3936],
        [4088,4395],
        [4652,4928],
        [5110,5317],
        [5639,5902],
        [6132,6531],
        [6845,7100],
        [7154,7401],
        [7701,7926]
    ],
    [
        [99,471],
        [641,984],
        [1220,1457],
        [1793,2011],
        [2259,2464],
        [2837,3036],
        [3114,3462],
        [3770,4022],
        [4361,4518],
        [4901,5050],
        [5206,5489],
        [5812,6007],
        [6376,6599],
        [7007,7113],
        [7205,7536],
        [7857,8079]
    ]
]

#print(f"{circulants[0][0]}")

H_list = [[0]*8176 for _ in range(1022)]
#print(f"{np.shape(H_list)}")


for i in range(2):
    for j in range(16):
        for position in circulants[i][j]:
            H_list[i*511][position] = 1
#print(f"{sum(H_list[511])}")
#print(f"H_list=\n{H_list}")


'''
for i in range(7154):
    for j in range(8176):
        if H_list[i][j] == 1:
            print(f"{i},{j}")
'''

for i in range(2):
    for j in range(16):
        for k in range(510): # row index is (i*511)+k
            for l in range(511): # column index is (j*511)+l
                H_list[(i*511)+k+1][(j*511)+((l+1)%511)] = H_list[(i*511)+k][(j*511)+l]


'''
PWD=os.path.dirname(os.path.realpath(__file__))

with open(PWD+'/circulants_for_H.txt', 'w') as file:
    for i in range(7154):
        for j in range(8176):
            file.write(str(H_list[i][j]))
        file.write("\n")
'''