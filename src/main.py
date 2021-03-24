import numpy as np 
import pandas as pd 
import os

import matplotlib.pyplot as plt 
from skimage import io, transform 
from tqdm.notebook import tqdm

import torch 
from torch.utils.data import Dataset 
from torchvision import transforms as T 

pd.options.mode.chained_assignment = None

df = pd.read_csv(r'dataset\train.csv')
pritn(df.head())


def get_APN(data):
    
    data['Positive'] = np.zeros(len(data))
    data['Negative'] = np.zeros(len(data))
    data = data.rename(columns = {'labels_group' : 'label_group','image' : 'Anchor'})
    
    for i in tqdm(range(len(data))):
        
        lg = data['label_group'].iloc[i]
        A_choice = data['Anchor'].iloc[i]
        img_id = data['posting_id'].loc[i]
        
        try : 
            P_choice = data['posting_id'][data['label_group'] == lg].values
        except: 
            P_choice = [img_id]
    
        if len(P_choice) == 1:
            data['Positive'].iloc[i] = data['Anchor'][data['posting_id'] == P_choice[0]].values[0]
            
        else:
            
            p_flag = True 
            n_flag = True
            
            while p_flag == True:
                
                P_choice = np.random.choice(data['posting_id'][data['label_group'] == lg].values)
                
                if img_id == P_choice:
                    p_flag = True
                    
                else:
                    data['Positive'].iloc[i] = data['Anchor'][data['posting_id'] == P_choice].values[0]
                    p_flag = False
                    
                
            while n_flag == True:
                
                N_choice = np.random.choice(data['Anchor'].values)
                
                if (N_choice == A_choice) or (N_choice == P_choice):
                    n_flag = True
                    
                else: 
                    data['Negative'].iloc[i] = N_choice
                    n_flag = False 
                    
    return data
