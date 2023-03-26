import torch 
from torchvision import transforms
from PIL import Image
import time

#import torch.nn as nn
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

import math

import copy
from typing import Optional, List

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn, Tensor

from torch.nn.utils.rnn import pad_sequence

from scipy.optimize import linear_sum_assignment

from timeit import default_timer as timer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_fn(batch_len,PAD_IDX,train=True,recon=False,run=12):
    #print('batch',len(batch),batch)
    src1_batch, src2_batch, y_batch,d_batch = [], [], [], []
    if train:
        E1,E2,A,D,run,t=loadgraph()
        #print(run,t)
        im1,im2=cnn_loader(run=run,t_r=t)
    elif recon:
        E1,E2,A,D,run,t=loadgraph(recon=True, train=False,run=run,t_r=j)
        im1,im2=cnn_loader(run=run,t_r=j)
        #print('recon')
    else:
        E1,E2,A,D,run,t=loadgraph(train=False)
        im1,im2=cnn_loader(run=run,t_r=t)
    #print('src_sample',src_sample)
    src1_batch.append(E1)
    #print('emb',src_batch[-1])
    src2_batch.append(E2)
    y_batch.append(A)
    d_batch.append(D)


    #print('src_batch',src1_batch[3])
    #print('src2_batch',src2_batch[3])
    #print('src_batch s',len(src_batch))
    src1_batch = pad_sequence(src1_batch, padding_value=PAD_IDX)
    #print('src_batch',src_batch)
    #print('src_batch s',src_batch.size())
    src2_batch = pad_sequence(src2_batch, padding_value=PAD_IDX)
    
    
    #print('src1',src1_batch[:,0,:],src1_batch[:,0,:].size())
    #print('src2',src2_batch[:,0,:],src2_batch[:,0,:].size())
    #print('y',y_batch)
    ##
    return src1_batch,src2_batch,im1,im2,y_batch,d_batch




def loadgraph(train=True,run=None,easy=False,recon=False,t_r=None):
    convert_tensor = transforms.ToTensor()
    if train:
        if run==None:
            run=np.random.randint(1,75) #!!!!!!!!!!##100 total data size
        else: run=run
        E=np.loadtxt('./'+str(run)+'/'+'embed.txt')
        #print('E',E.shape)
        id,tt = np.loadtxt('./'+str(run)+'/'+'timetable.txt', delimiter='\t', usecols=(0,1), unpack=True)
        A=np.loadtxt('./'+str(run)+'_GT'+'/'+'A.txt')
        D=np.loadtxt('./'+str(run)+'/'+'D.txt')
        bg=A[0]
        for i in range(len(A)):
            for j in range(len(A)):
                if i>j:
                    A[i,j]=0
        #A=A+np.eye(len(A), dtype=int)
        
        t = np.random.randint(30) #!!!!!!!!how many t??
        id1 = id[tt==t].astype(int)
        if t==0:
            id1=id1[1:]
        id2 = id[tt==(t+1)].astype(int)
        
        #print(run,t,id1,id2)
        
        E1 = E[id1-1]
        E2 = E[id2-1]
       
        E_bg = E[0]
        
        E1=np.concatenate((np.array([E_bg]), E1), axis=0)
        E2=np.concatenate((np.array([E_bg]), E2), axis=0)
        
     
        
        A=A[id1-1]
        
        A=A[:,id2-1]
        
        
        D=D[id1-1]
        D=D[:,id2-1]
        
        
        
       
        
        #print(bg[id1-1])
        #print(bg[id2-1])
        
        
        A=np.concatenate((np.array([bg[id2-1]]), A), axis=0)
        
        bg_a=np.append(1,bg[id1-1])
        #print(bg_a)
        A=np.concatenate((np.array([bg_a]).T, A), axis=1)
        
        bg_b = np.append(0,np.zeros(len(bg[id1-1])))
        
        D=np.concatenate((np.array([np.zeros(len(bg[id2-1]))]), D), axis=0)
        D=np.concatenate((np.array([bg_b]).T, D), axis=1)
        
        #print(D)
        #print(np.dot(E1,E2.T))
        
        
        
        
    else:
        #print('eval')
        if run==None:
            run=np.random.randint(75,100) #!!!!!!!!
        else: run=run
        E=np.loadtxt('./'+str(run)+'/'+'embed.txt')
        id,tt = np.loadtxt('./'+str(run)+'/'+'timetable.txt', delimiter='\t', usecols=(0,1), unpack=True)
        A=np.loadtxt('./'+str(run)+'_GT'+'/'+'A.txt')
        D=np.loadtxt('./'+str(run)+'/'+'D.txt')
        bg=A[0]
        for i in range(len(A)):
            for j in range(len(A)):
                if i>j:
                    A[i,j]=0
        #A=A+np.eye(len(A), dtype=int)
        
        t = np.random.randint(30) #!!!!!!!!how many t??
        id1 = id[tt==t].astype(int)
        if t==0:
            id1=id1[1:]
        id2 = id[tt==(t+1)].astype(int)
        
        #print(run,t,id1,id2)
        
        E1 = E[id1-1]
        E2 = E[id2-1]
       
        E_bg = E[0]
        
        E1=np.concatenate((np.array([E_bg]), E1), axis=0)
        E2=np.concatenate((np.array([E_bg]), E2), axis=0)
        
     
        
        A=A[id1-1]
        
        A=A[:,id2-1]
        
        #print(A)
        
        D=D[id1-1]
        D=D[:,id2-1]
        
       
        
        #print(bg[id1-1])
        #print(bg[id2-1])
        
        
        A=np.concatenate((np.array([bg[id2-1]]), A), axis=0)
        
        bg_a=np.append(1,bg[id1-1])
        A=np.concatenate((np.array([bg_a]).T, A), axis=1)
        
        bg_b = np.append(0,np.zeros(len(bg[id1-1])))
        
        D=np.concatenate((np.array([np.zeros(len(bg[id2-1]))]), D), axis=0)
        D=np.concatenate((np.array([bg_b]).T, D), axis=1)
        
        
    if recon: 
        run=run
        E=np.loadtxt('./'+str(run)+'/'+'embed.txt')
        id,tt = np.loadtxt('./'+str(run)+'/'+'timetable.txt', delimiter='\t', usecols=(0,1), unpack=True)
        A=np.loadtxt('./'+str(run)+'_GT'+'/'+'A.txt')
        D=np.loadtxt('./'+str(run)+'/'+'D.txt')
        for i in range(len(A)):
            for j in range(len(A)):
                if i>j:
                    A[i,j]=0
        #A=A+np.eye(len(A), dtype=int)
        
        
        #print(id)
        t = t_r
        id1 = id[tt==t].astype(int)
        if t==0:
            id1=id1[1:]
        id2 = id[tt==(t+1)].astype(int)
        
        #print(run,t,id1,id2)
        
        E1 = E[id1-1]
        E2 = E[id2-1]
       
        E_bg = E[0]
        
        E1=np.concatenate((np.array([E_bg]), E1), axis=0)
        E2=np.concatenate((np.array([E_bg]), E2), axis=0)
        
     
        
        A=A[id1-1]
        
        A=A[:,id2-1]
        
        
        
        D=D[id1-1]
        D=D[:,id2-1]
        
        
        #print(A)
        
        
        #print(bg[id1-1])
        #print(bg[id2-1])
        
        
        A=np.concatenate((np.array([bg[id2-1]]), A), axis=0)
        
        bg_a=np.append(1,bg[id1-1])
        A=np.concatenate((np.array([bg_a]).T, A), axis=1)
       
        #print(E1,E2)
        
        bg_b = np.append(0,np.zeros(len(bg[id1-1])))
    
    
    
        D=np.concatenate((np.array([np.zeros(len(bg[id2-1]))]), D), axis=0)
        D=np.concatenate((np.array([bg_b]).T, D), axis=1)
    
    
    
   
    
    D=D.astype(np.float32)
    
    vd = np.vectorize(d_mask_function,otypes=[float])
    
    D = vd(D,0.15,-2.0)
    
    
    E1=E1.astype(np.float32)
    E2=E2.astype(np.float32)
    A=A.astype(np.float32)
    #A=A.astype(np.float32)
    
    
    
    E1=convert_tensor(E1) 
    E2=convert_tensor(E2) 
    A=convert_tensor(A)
    D=convert_tensor(D)
    
    #print(E1[0].size(),E1[0])
    #print(E2[0].size(),E2[0])
    #print(A,A.size())
    #print('E',E.size())
    
    return E1[0],E2[0],A[0],D[0],run,t

def create_mask(src,PAD_IDX):
    
    src= src[:,:,0]

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    #print('src_padding_mask',src_padding_mask,src_padding_mask.size())
    return src_padding_mask


class makeAdja:
    def __init__(self):
        pass
        
    def forward(self,z:Tensor,
                mask1: Tensor,
                mask2: Tensor):
        Ad = []
        for i in range(z.size(0)):
            n=len([i for i, e in enumerate(mask1[i]) if e != True])
            m=len([i for i, e in enumerate(mask2[i]) if e != True])
            Ad.append(z[i,0:n,0:m])
        
        
        return Ad
    
def cnn_loader(run=None,t_r=None):


    run=run
    t=t_r
    E1=torch.load('./'+str(run)+'/'+str(t)+'.pt')
    E2=torch.load('./'+str(run)+'/'+str(t+1)+'.pt')

    
   

    
    #print(E1[0].size(),E1[0])
    #print(E2[0].size(),E2[0])
    #print('A',A.size())
    #print('E1',E1.size())
    #print('E2',E2.size())
    
    return E1,E2   
    
def train_epoch(model, optimizer,loss_fn):
    model.train()
    losses = 0
    
    src1, src2,im1,im2,y,d = collate_fn(1,-100)
        
    src1= src1.to(DEVICE)
    src2= src2.to(DEVICE)
    
    im1= im1.to(DEVICE)
    im2= im2.to(DEVICE)
    
    src_padding_mask1=create_mask(src1,-100)
    src_padding_mask2=create_mask(src2,-100)
    try:
        Ad,out1,out2,out_dec1,src1_t1,src2_t2 = model(src1,src2,src_padding_mask1,src_padding_mask2,im1,im2)
    except:    
        Ad = model(src1,src2,src_padding_mask1,src_padding_mask2,im1,im2)

    optimizer.zero_grad()

   
    loss = loss_fn.loss(Ad,y)
    
    #print(Ad[0],y[0])
    #print('l',loss)
    #print('l',loss.item() / len(src1))
    
    loss.backward()

    optimizer.step()
    losses += loss.item()
    
    

    return losses / len(src1)





class Loss():
    def __init__(self,pen):
        self.pen=pen
        
    def loss (self,Ad,y):
        
        loss=0
        
        for i in range(len(Ad)):
            l = nn.CrossEntropyLoss()
            
            y[i] = y[i].to(DEVICE)
            
            s = l(Ad[i], y[i])
            
            loss=loss+s
                
        
        return loss
    


def evaluate(model,loss_fn):
    model.eval()
    losses = 0

    src1, src2,im1,im2,y,d = collate_fn(1,-100)
        
    src1= src1.to(DEVICE)
    src2= src2.to(DEVICE)
    
    im1= im1.to(DEVICE)
    im2= im2.to(DEVICE)
    
    src_padding_mask1=create_mask(src1,-100)
    src_padding_mask2=create_mask(src2,-100)
    try:
        Ad,out1,out2,out_dec1,src1_t1,src2_t2 = model(src1,src2,src_padding_mask1,src_padding_mask2,im1,im2)
    except:    
        Ad = model(src1,src2,src_padding_mask1,src_padding_mask2,im1,im2)

    
   
    loss = loss_fn.loss(Ad,y)
    
    losses += loss.item()
    
        

    return losses / len(src1)



def square(m):
    return m.shape[0] == m.shape[1]


def d_mask_function(x,r_core,alpha):
    if x < r_core:
        return 1
    else:
        return (x/r_core)**alpha


class AdjTra_cnn(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 out = False, 
                 dim_feedforward: int = 512,
                 dropout: float = 0.05):
        super(AdjTra_cnn, self).__init__()
        
        
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead,dim_feedforward=dim_feedforward)
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_size, nhead=nhead,dim_feedforward=dim_feedforward)
        
        
        self.decoder_1 = nn.TransformerDecoder(decoder_layer, num_layers=num_encoder_layers)
        self.decoder_2 = nn.TransformerDecoder(decoder_layer, num_layers=num_encoder_layers)
        #self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        
        self.out=out 

        
        self.conv1_1 = nn.Conv2d(3, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(64, 125, 3)
        
   
        self.fc1_1 = nn.Sequential(
            nn.Linear(1125, 350),
            nn.LeakyReLU())
        
        self.fc2_1 = nn.Sequential(
            nn.Linear(350, 60),
            nn.LeakyReLU())
        
       
        
        self.conv1_2 = nn.Conv2d(3, 64, 3)
        self.conv2_2 = nn.Conv2d(64, 125, 3)
        
        self.fc1_2 = nn.Sequential(
            nn.Linear(1125, 350),
            nn.LeakyReLU())
        
        self.fc2_2 = nn.Sequential(
            nn.Linear(350, 60),
            nn.LeakyReLU())

    
        self.drop = nn.Dropout(p=dropout)


        
        self.sig = torch.nn.Sigmoid()
        self.Ad = makeAdja()
        

    def forward(self,
                src_t1: Tensor,
                src_t2: Tensor,
                src_padding_mask1: Tensor,
                src_padding_mask2: Tensor,
                im_src1: Tensor,
                im_src2: Tensor):
        
        #print('trans_src_before_pos',src_t1,src_t1.size())
        #print('trans_src_toke',self.src_tok_emb(src),self.src_tok_emb(src).size())
      
        
        src1_emb = src_t1[:,0,:]
        src2_emb = src_t2[:,0,:]
        #print('src1',src1_emb.size())
        #print('src2',src2_emb.size())
        
        im_t1 = self.pool(F.relu(self.conv1_1(im_src1)))
        im_t1 = self.pool(F.relu(self.conv2_1(im_t1)))
        
        im_t1 = torch.flatten(im_t1, 1)
        im_t1 = self.drop(im_t1)
        
        #print('sizes1',im_t1.size())
        
        im_t1 = self.fc1_1(im_t1)
        im_t1 = self.fc2_1(im_t1)
    
        
        
        
        im_t2 = self.pool(F.relu(self.conv1_2(im_src2)))
        im_t2 = self.pool(F.relu(self.conv2_2(im_t2)))
        
        im_t2 = torch.flatten(im_t2, 1)
        im_t2 = self.drop(im_t2)
        
        #print('sizes1',src_t1.size())
        
        im_t2 = self.fc1_1(im_t2)
        im_t2 = self.fc2_1(im_t2)
        
        
        #print('sizes1',src1_emb.size(),im_t1.size())
        #print('sizes2',src2_emb.size(),im_t2.size())
        
        
        
        out_dec1=self.decoder_1(src1_emb, im_t1)
        
        out_dec2=self.decoder_2(src2_emb, im_t2)
     
        
        #out_dec1=torch.transpose(out_dec1,0,1)
        out_dec2=torch.transpose(out_dec2,0,1)
        
        
        z=self.sig(torch.matmul(out_dec1,out_dec2))
        
        #Ad=self.Ad.forward(z,src_padding_mask1,src_padding_mask2)


        return [z]


    
    
    
    
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
    

emb_size= 60
nhead= 6
num_encoder_layers = 3


transformer = AdjTra_cnn(num_encoder_layers, emb_size, nhead)
#transformer = AdjacencyTransformer(num_encoder_layers, emb_size, nhead)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = Loss(pen=0)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)





NUM_EPOCHS = 350

loss_over_time=[]
test_error=[]
epoch_mean_test=[]
epoch_mean_test=[]
epoch_mean_train=[]


for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer,loss_fn)
    epoch_mean_train.append(train_loss)
    end_time = timer()
    val_loss = evaluate(transformer,loss_fn)
    epoch_mean_test.append(val_loss)
    if epoch % 100 == 0:
        test_error.append(np.mean(epoch_mean_test))
        loss_over_time.append(np.mean(epoch_mean_train))
        
    
        #loss_over_time.append(train_loss)
        np.savetxt('./'+'train_loss_AttTrack_cnn.txt', np.c_[loss_over_time],delimiter='\t',header='trainloss')
        
        #test_error.append(val_loss)
                    
        np.savetxt('./'+'test_loss_AttTrack_cnn.txt', np.c_[test_error],delimiter='\t',header='testloss')
        
        if test_error[-1]==np.min(test_error):
            torch.save(transformer.state_dict(), 'AttTrack_cnn.pt')
            
        epoch_mean_test=[]
        epoch_mean_train=[]
    if epoch < 25:
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

#torch.save(transformer.state_dict(), 'AttTrack_2.pt')
