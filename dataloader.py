import torch
import numpy as np
import pickle
import random

from torch.utils.data import Dataset, DataLoader

train_data_file = "./train_data/train_data.pkl"
val_data_file = "./train_data/val_data.pkl"

class RoadDataSet(Dataset):
    def __init__(self, datafile, n_sample):
        '''
        n_sample: the number of 15min time segments, 96 means 24 hrs' segments. 
        '''
        super().__init__()
        with open(datafile, 'rb') as f:
            data = pickle.load(f)
        self.train_raw=data['speed_ary']
        self.geo_nebor_idx=data['geo_nebor_idx']
        self.train_time_feature=data['time_feature']
        self.n_road_seg=self.train_raw.shape[0]
        self.n_time_seg=self.train_raw.shape[1]-24 #-24 because need reserve 6hr time seg for target.
        self.n_sample=n_sample

    def __len__(self):
        return(2000*(self.n_time_seg-self.n_sample))

    def __getitem__(self, idx):
        rid=idx // (self.n_time_seg-self.n_sample)
        tid=idx % (self.n_time_seg-self.n_sample)
        a= self.train_raw[rid, tid:tid+self.n_sample] # main speed
        b= self.train_raw[self.geo_nebor_idx[rid][:-1],tid:tid+self.n_sample] #neibor speed
        src_seq=np.concatenate((b.T,a.reshape((-1,1))),axis=1) 
        c=self.train_time_feature[tid:tid+self.n_sample] #time feature for all samples, busy hour, work day, etc.
        src_seq=np.concatenate((src_seq,c),axis=1) 
        
        
        tgt1=self.train_raw[rid,tid+self.n_sample+1]
        tgt2=self.train_raw[rid,tid+self.n_sample+4]
        tgt3=self.train_raw[rid,tid+self.n_sample+24]
        return src_seq, tgt1, tgt2, tgt3

class RoadDataSet2(Dataset):
    def __init__(self, datafile, n_sample, vol):
        '''        
        Structure:
        0:n_sample:                               speed for current point in n_sample time
        n_sample:n_sample+1:             speed for current point in 24hr ago   
        n_sample+1:n_sample+1+44:  speed for neighbor point
        left:                                            speed for neighbor points for last 2 time seg
        '''
        super().__init__()
        with open(datafile, 'rb') as f:
            data = pickle.load(f)
        self.train_raw=data['speed_ary']
        self.geo_nebor_idx=data['geo_nebor_idx']
        self.mask=data['mask']
        self.n_road_seg=self.train_raw.shape[0]
        self.n_time_seg=self.train_raw.shape[1]-24 #-24 because need reserve 6hr time seg for target.
        self.n_sample=n_sample
        self.n_max=(self.n_road_seg-1)*(self.n_time_seg-96) - 1
        self.steps=vol

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        idx=random.randint(0,self.n_max)
        rid=idx // (self.n_time_seg-96)
        tid=idx % (self.n_time_seg-96)
        a = self.train_raw[rid, tid+96-self.n_sample:tid+96] # main speed
        b = self.train_raw[rid, tid].reshape((1,))  # speed 24hr ago 
        c = self.train_raw[self.geo_nebor_idx[rid],tid+96-3:tid+96] #neibor speed in 3 time seg
        src_seq=np.concatenate((a,b,c.reshape(-1))) 
        
        m=np.zeros((self.n_sample+1,))
        mask=np.concatenate((m, self.mask[rid], self.mask[rid], self.mask[rid]))
        
        tgt1=self.train_raw[rid,tid+96+1]
        tgt2=self.train_raw[rid,tid+96+4]
        tgt3=self.train_raw[rid,tid+96+24]
        return np.expand_dims(src_seq,axis=2), mask, tgt1, tgt2, tgt3

class RoadDataSet3(Dataset):
    def __init__(self, datafile, n_sample, vol):
        '''        
        Structure:
        0:n_sample:                               speed for current point in n_sample time
        n_sample:n_sample+1:             speed for current point in 24hr ago   
        n_sample+1:n_sample+1+44:  speed for neighbor point
        left:                                            speed for neighbor points for last 2 time seg
        '''
        super().__init__()
        with open(datafile, 'rb') as f:
            data = pickle.load(f)
        self.train_raw=data['speed_ary']
        self.geo_nebor_idx=data['geo_nebor_idx']
        self.time_feature=data['time_feature']
        self.mask=data['mask']
        self.gps=data['gps']
        self.attrs=data['link_attrs']
        self.n_road_seg=self.train_raw.shape[0]
        self.n_time_seg=self.train_raw.shape[1]-24 #-24 because need reserve 6hr time seg for target.
        self.n_sample=n_sample
        self.n_max=(self.n_road_seg-1)*(self.n_time_seg-96) - 1
        self.steps=vol

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        idx=random.randint(0,self.n_max)
        rid=idx // (self.n_time_seg-96)
        tid=idx % (self.n_time_seg-96)
        t = self.time_feature[tid+96,0:4]
        g = self.gps[rid]
        at= self.attrs[rid]
        a = self.train_raw[rid, tid+96-self.n_sample:tid+96] # main speed
        b = self.train_raw[rid, tid].reshape((1,))  # speed 24hr ago 
        c = self.train_raw[self.geo_nebor_idx[rid],tid+96-3:tid+96] #neibor speed in 3 time seg
        src_seq=np.concatenate((t,g,at,a,b,c.reshape(-1))) 
        
        m=np.zeros((4+2+21+self.n_sample+1,))
        mask=np.concatenate((m, self.mask[rid], self.mask[rid], self.mask[rid]))
        
        tgt1=self.train_raw[rid,tid+96+1]
        tgt2=self.train_raw[rid,tid+96+4]
        tgt3=self.train_raw[rid,tid+96+24]
        return np.expand_dims(src_seq,axis=2), mask, tgt1, tgt2, tgt3

class RoadDataSet4(Dataset):
    def __init__(self, datafile, n_sample, vol):
        '''
        batch x 45 x features.
        features including n_sample speed fc in time seg, road attrs, etc.
        '''
        super().__init__()
        with open(datafile, 'rb') as f:
            data = pickle.load(f)
        self.train_raw=data['speed_ary']
        self.geo_nebor_idx=data['geo_nebor_idx']
        self.time_feature=data['time_feature']
        self.attrs=data['link_attrs']
        self.mask=data['mask']
        self.n_road_seg=self.train_raw.shape[0]
        self.n_time_seg=self.train_raw.shape[1]-24-96 #-24 because need reserve 6hr time seg for target.\
                                                      #-96 reserve for input data time range.
        self.n_sample=n_sample
        self.n_max=(self.n_road_seg-1)*(self.n_time_seg-96) - 1
        self.steps=vol

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        idx=random.randint(0,self.n_max)
        rid=idx // (self.n_road_seg)
        tid=idx % (self.n_time_seg-self.n_sample)
        all=np.append(np.asarray(rid),self.geo_nebor_idx[rid]) #id and it's neigbours
        
        a=self.train_raw[all, tid+96-self.n_sample:tid+96]
        b=self.train_raw[all, tid].reshape((-1,1))  # speed 24hr ago 
        c=self.attrs[all]  #attributes like lane number, length, rank value.
        src_seq=np.concatenate((a,b,c),axis=1) 
        m=np.zeros((1,))
        mask=np.concatenate((m, self.mask[rid]))
        tgt1=self.train_raw[rid,tid+96+1]
        tgt2=self.train_raw[rid,tid+96+4]
        tgt3=self.train_raw[rid,tid+96+24]
        return src_seq, mask,tgt1, tgt2, tgt3
