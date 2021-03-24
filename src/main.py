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
print(df.head())


def get_APN(data):
    
    data['Positive'] = np.zeros(len(data))
    data['Negative'] = np.zeros(len(data))
    data = data.rename(columns = {'labels_group' : 'label_group','image' : 'Anchor'})
    
    for i in range(len(data)):
        
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

df_siamese = get_APN(df)

df_siamese = df_siamese.drop(['label_group','posting_id','image_phash','title'],axis = 1)

class APN_Dataset(Dataset):
    
    def __init__(self,df,root_dir,transform = None):
        
        self.APN_names = df
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.ANP_names)
    
    def __getitem__(self,idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        A_name = os.path.join(self.root_dir,self.APN_names.iloc[idx,0])
        P_name = os.path.join(self.root_dir,self.APN_names.iloc[idx,1])
        N_name = os.path.join(self.root_dir,self.APN_names.iloc[idx,2])
        
        A = io.imread(A_name)
        P = io.imread(P_name)
        N = io.imread(N_name)
         
        
        if self.transform:
            sample = self.transform((A,P,N))
            A,P,N = sample
            
        return A,P,N


class ToTensor(object):
    
    def __call__(self,sample):
        
        A,P,N = sample
        
        A = A.transpose((2,0,1))
        P = P.transpose((2,0,1))
        N = N.transpose((2,0,1))
        
        return torch.from_numpy(A),torch.from_numpy(P),torch.from_numpy(N)

class Resize(object):
    
    def __init__(self,*img_size):
        self.img_size = img_size
    
    def __call__(self,sample):
        
        A,P,N = sample
        
        A = transform.resize(A,self.img_size)
        P = transform.resize(P,self.img_size)
        N = transform.resize(N,self.img_size)
        
        return A,P,N
    

custom_transform = T.Compose([
    Resize(512,512),
    ToTensor()
])

siamese_data = APN_Dataset(df_siamese,'dataset/train_images',transform = custom_transform)

idx = 3245
A,P,N = siamese_data[idx]

f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize= (10,5))

ax1.set_title('Anchor')
ax1.imshow(A.numpy().transpose((1,2,0)), cmap = 'gray')

ax2.set_title('Positive')
ax2.imshow(P.numpy().transpose((1,2,0)), cmap = 'gray')

ax3.set_title('Negative')
ax3.imshow(N.numpy().transpose((1,2,0)), cmap = 'gray')


idx = 9005
A,P,N = siamese_data[idx]

f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize= (10,5))

ax1.set_title('Anchor')
ax1.imshow(A.numpy().transpose((1,2,0)), cmap = 'gray')

ax2.set_title('Positive')
ax2.imshow(P.numpy().transpose((1,2,0)), cmap = 'gray')

ax3.set_title('Negative')

ax3.imshow(N.numpy().transpose((1,2,0)), cmap = 'gray')

idx = 23380
A,P,N = siamese_data[idx]

f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize= (10,5))

ax1.set_title('Anchor')
ax1.imshow(A.numpy().transpose((1,2,0)), cmap = 'gray')

ax2.set_title('Positive')
ax2.imshow(P.numpy().transpose((1,2,0)), cmap = 'gray')

ax3.set_title('Negative')
ax3.imshow(N.numpy().transpose((1,2,0)), cmap = 'gray')

df_siamese.to_csv('APN_data.csv',index = False)