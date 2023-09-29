'''

Author: Enneng Yang (ennengyang@gmail.com)
function:
    
'''
import sys
import os
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import pandas as pd
datas = pd.read_csv(r'C:\Users\Administrator\Desktop\EuroSAT\train.csv',usecols=['Filename'])
train = datas.values.tolist()
datas = pd.read_csv(r'C:\Users\Administrator\Desktop\EuroSAT\test.csv',usecols=['Filename'])
test = datas.values.tolist()
datas = pd.read_csv(r'C:\Users\Administrator\Desktop\EuroSAT\validation.csv',usecols=['Filename'])
validation = datas.values.tolist()



for img in test:
    img = img[0]
    fold = r'C:\Users\Administrator\Desktop\EuroSAT'+'\\test\\'+img.split('/')[0]
    if not os.path.exists(fold):
        os.makedirs(fold)
    shutil.move(r'C:\Users\Administrator\Desktop\EuroSAT\\'+img, r'C:\Users\Administrator\Desktop\EuroSAT'+'\\test\\'+img)