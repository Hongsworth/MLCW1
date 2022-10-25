import numpy as np

f_arr = []

#for best gain maybe compare each split and averagin all 4 class gain? then find the best split

#store all possible splits in an array?

def FIND_SPLIT_S(f_arr,room,col):
    b_gain = 0
    b_split = 0
    c = [0,0,0,0]
    room = len(s_arr[0])
    l = len(s_arr)
    for col in range(len(s_arr[0]-1)):
        s_arr = f_arr[f_arr[:, col].argsort()]
        for i in range(len(s_arr)):
            if (s_arr[i][col] != s_arr[i+1][col]):
                c_split = ((s_arr[i][col] + s_arr[i+1][col])/2)
                for j in range(len(s_arr)):
                    if (s_arr[i][col] < c_split):
                        c[room] = c[room]+1
                c_gain = calc_entropy(c,l)
                if (c_gain > b_gain):
                    b_gain = c_gain
                    b_col = col
                    b_split = c_split
    return b_col, b_split


def calc_entropy(c,l):
    c_gain = 0
    for n in range(len(c)):
        c_gain += (-(c[n]/l)-(-c[n]/l))
    return c_gain
