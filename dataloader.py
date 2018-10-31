import torch
import numpy as np
import pickle
import random

from torch.utils.data import Dataset

class RoadDataSetBase(Dataset):  
    def __init__(self, datafile, n_sample, steps, phase):
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
        self.gps=data['gps']
        self.attrs=data['link_attrs']
        self.mask=data['mask']
        self.n_sample=n_sample
        self.n_road_seg=self.train_raw.shape[0]-1  #-1 becuase the padding ling
        self.n_time_seg=self.train_raw.shape[1]-24-self.span #-24 because need reserve 6hr time seg for target.\
                                                             #-span reserve for input data time span.
        self.n_max=(self.n_road_seg)*(self.n_time_seg)
        self.steps=steps
        self.phase=phase
        
    @property
    def span(self):
        if self.n_sample>96:
            return self.n_sample
        else:
            return 96
    
    def __len__(self):
        if self.phase=='test':
            return self.n_max
        else:
            return self.steps

    def __getitem__(self, idx):
        """
        override this
        """
        pass

    def get_random_id(self):
        if self.phase == "train":
            max=int(self.n_road_seg*0.8)
            return random.randint(0,max-1), random.randint(0,self.n_time_seg-1)
        elif self.phase == "val":
            min=int(self.n_road_seg*0.8)
            return random.randint(min,self.n_road_seg-1), random.randint(0,self.n_time_seg-1)
    


class RoadDataSet3(RoadDataSetBase):
    def __init__(self, datafile, n_sample, steps, phase):
        '''        
        batch x features x 1
        
        '''
        super().__init__(datafile, n_sample, steps, phase)
    
    def __getitem__(self, idx):
        if self.phase=='test':
            rid=idx // (self.n_time_seg)
            #rid=1
            tid=idx % (self.n_time_seg)
        else:
            rid,tid=self.get_random_id()
        cur=tid+self.span #current time point
        t = self.time_feature[cur,0:4]
        g = self.gps[rid]
        at= self.attrs[rid]
        a = self.train_raw[rid, cur-self.n_sample:cur] # main speed
        b = self.train_raw[rid, tid].reshape((1,))  # speed 24hr ago 
        c = self.train_raw[self.geo_nebor_idx[rid],cur-3:cur] #neibor speed in 3 time seg
        src_seq=np.concatenate((t,g,at,a,b,c.reshape(-1))) 
        
        m=np.zeros((4+2+21+self.n_sample+1,))
        mask=np.concatenate((m, self.mask[rid], self.mask[rid], self.mask[rid]))
        
        tgt1=self.train_raw[rid,cur+1]
        tgt2=self.train_raw[rid,cur+4]
        tgt3=self.train_raw[rid,cur+24]
        return np.expand_dims(src_seq,axis=2), mask, tgt1, tgt2, tgt3

class RoadDataSet4(RoadDataSetBase):
    def __init__(self, datafile, n_sample, steps, phase):
        '''
        batch x 45 x features.
        features including n_sample speed fc in time seg, road attrs, etc.
        '''
        super().__init__(datafile, n_sample, steps, phase)

    def __getitem__(self, idx):
        if self.phase=='test':
            rid=idx // (self.n_time_seg)
            tid=idx % (self.n_time_seg)
        else:
            rid,tid=self.get_random_id()
        all=np.append(np.asarray(rid),self.geo_nebor_idx[rid]) #id and it's neigbours
        cur=tid+self.span #current time point
        a=self.train_raw[all, cur-self.n_sample:cur]
        b=self.train_raw[all, tid].reshape((-1,1))  # speed 24hr ago 
        c=self.attrs[all]  #attributes like lane number, length, rank value.
        src_seq=np.concatenate((a,b,c),axis=1) 
        m=np.zeros((1,))
        mask=np.concatenate((m, self.mask[rid]))
        tgt1=self.train_raw[rid,cur+1]
        tgt2=self.train_raw[rid,cur+4]
        tgt3=self.train_raw[rid,cur+24]
        return src_seq, mask,tgt1, tgt2, tgt3

    
    
    
    
    
    
class _RoadDataSet2(Dataset):
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
        self.n_road_seg=self.train_raw.shape[0]-1  #-1 becuase the padding ling
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