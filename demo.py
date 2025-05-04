import pandas as pd
import scipy.io
import numpy as np
def func1(cnt):
    df = pd.read_csv("C:/Users/Lenovo/Desktop/Projection/"+str(cnt)+".csv", encoding='utf-8',header=None)

    df = df.drop([2, 5,8,11], axis=1)

    df.columns = ['inx','iny','indirx','indiry','outx','outy','outdirx','outdiry','wepl']

    # 创建Python字典（模拟MATLAB struct）
    data = {
        'data': {
            'Wepl': df['wepl'].values.reshape(-1,1),
            'dirIn': np.stack((df['indirx'].values, df['indiry'].values)).T.flatten().reshape(-1,1),
            'dirOut': np.stack((df['outdirx'].values, df['outdiry'].values)).T.flatten().reshape(-1,1),
            'posIn': np.stack((df['inx'].values, df['iny'].values)).T.flatten().reshape(-1,1),
            'posOut':   np.stack((df['outx'].values, df['outy'].values)).T.flatten().reshape(-1,1),
        }
    }

    scipy.io.savemat("proton/"+str(cnt)+'.mat', data)

for i in range(0,72):
    func1(i*5)