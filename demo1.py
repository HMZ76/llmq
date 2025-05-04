import pandas as pd
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def read_df(num):
    df = pd.read_csv("C:/Users/Lenovo/Desktop/Projection/"+str(num)+".csv", encoding='utf-8',header=None)

    df = df.drop([2, 5,8,11], axis=1)

    df.columns = ['inx','iny','indirx','indiry','outx','outy','outdirx','outdiry','wepl']

    I = np.zeros((256,256,1))
    cnt = np.zeros((256,256,1))
    for value in df.values:
        v = value[8]
        I[int(value[0]),int(value[1]),0] += v
        cnt[int(value[0]),int(value[1]),0] += 1
    cnt[cnt==0] = 1
    I = I/(cnt)

    I =  (1-I/I.max())*255
   
    np.savetxt('data.csv', I[:,:,0], delimiter=',')
    return I[:,:,0] 
proj = []

for i in range(0,72):
    I = read_df(i*5)
    proj.append(I)
    break
proj = np.array(proj).transpose(1,2,0)

projections = {'projections': proj}
scipy.io.savemat('output.mat', projections)
print(proj)