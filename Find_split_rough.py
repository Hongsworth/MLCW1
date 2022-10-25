import numpy as np

def entropy(x, y):
    for i in range(7):
        for j in range(4):
            


def FIND_SPLIT(x,y,room,col):
    b_gain = 0
    for j in range(100):
        for i in range(len(x)):
            if x[i][col] < -j and y[i] == room:
                count += 1
        v = count/2000
        c_gain = -v * np.log2(v) - v * np.log2(v)
        if c_gain > b_gain:   
            b_gain = c_gain
            b_split = j    
    return b_split

