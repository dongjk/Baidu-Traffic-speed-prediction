import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ModelWrapper():
    def __init__(self, name, opt):
        self.model=eval(name)(opt)

class SimpleFC(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.fc1=nn.Linear(opt.n_max_seq*opt.d_model,opt.d_inner)
        self.dropout=nn.Dropout(opt.dropout)
        self.head1=nn.Linear(opt.d_inner,1)
        self.head2=nn.Linear(opt.d_inner,1)
        self.head3=nn.Linear(opt.d_inner,1)
    def forward(self,x,**kw):
        x=x.view(x.shape[0],-1)
        x=self.fc1(x)
        x=self.dropout(F.relu(x))
        h1=self.head1(x) #15 min
        h2=self.head2(x) #60 min
        h3=self.head3(x) #6 hrs
        return h1,h2,h3
        
class SimpleFC2(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.fc1=nn.Linear(opt.n_max_seq*opt.d_model,opt.d_inner)
        self.dropout=nn.Dropout(opt.dropout)
        self.fc2=nn.Linear(opt.d_inner,opt.d_inner2)
        self.head1=nn.Linear(opt.d_inner2,1)
        self.head2=nn.Linear(opt.d_inner2,1)
        self.head3=nn.Linear(opt.d_inner2,1)
    def forward(self,x,**kw):
        x=x.view(x.shape[0],-1)
        x=self.fc1(x)
        x=self.dropout(F.relu(x))
        x=self.fc2(x)
        x=F.relu(x)
        h1=self.head1(x) #15 min
        h2=self.head2(x) #60 min
        h3=self.head3(x) #6 hrs
        return h1,h2,h3

class SimpleFC3(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.fc1=nn.Linear(opt.n_max_seq*opt.d_model,opt.d_inner)
        self.dropout=nn.Dropout(opt.dropout)
        self.fc2=nn.Linear(opt.d_inner,opt.d_inner2)
        self.fc3=nn.Linear(opt.d_inner2,opt.d_inner3)
        self.head1=nn.Linear(opt.d_inner3,1)
        self.head2=nn.Linear(opt.d_inner3,1)
        self.head3=nn.Linear(opt.d_inner3,1)
    def forward(self,x,**kw):
        x=x.view(x.shape[0],-1)
        x=self.fc1(x)
        x=self.dropout(F.relu(x))
        x=self.fc2(x)
        x=F.relu(x)
        x=self.fc3(x)
        x=F.relu(x)
        h1=self.head1(x) #15 min
        h2=self.head2(x) #60 min
        h3=self.head3(x) #6 hrs
        return h1,h2,h3
    
class SimpleFC4(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.fc1=nn.Linear(opt.n_max_seq*opt.d_model,opt.d_inner)
        self.dropout=nn.Dropout(opt.dropout)
        self.fc2=nn.Linear(opt.d_inner,opt.d_inner2)
        self.fc3=nn.Linear(opt.d_inner2,opt.d_inner3)
        self.fc4=nn.Linear(opt.d_inner3,opt.d_inner4)
        self.head1=nn.Linear(opt.d_inner4,1)
        self.head2=nn.Linear(opt.d_inner4,1)
        self.head3=nn.Linear(opt.d_inner4,1)
    def forward(self,x,**kw):
        x=x.view(x.shape[0],-1)
        x=self.fc1(x)
        x=self.dropout(F.relu(x))
        x=self.fc2(x)
        x=F.relu(x)
        x=self.fc3(x)
        x=F.relu(x)
        x=self.fc4(x)
        x=F.relu(x)
        h1=self.head1(x) #15 min
        h2=self.head2(x) #60 min
        h3=self.head3(x) #6 hrs
        return h1,h2,h3   

    
class SimpleFC5(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.fc1=nn.Linear(opt.n_max_seq*opt.d_model,opt.d_inner)
        self.dropout=nn.Dropout(opt.dropout)
        self.fc2=nn.Linear(opt.d_inner,opt.d_inner2)
        self.fc3=nn.Linear(opt.d_inner2,opt.d_inner3)
        self.fc4=nn.Linear(opt.d_inner3,opt.d_inner4)
        self.fc5=nn.Linear(opt.d_inner4,opt.d_inner5)
        self.head1=nn.Linear(opt.d_inner5,1)
        self.head2=nn.Linear(opt.d_inner5,1)
        self.head3=nn.Linear(opt.d_inner5,1)
    def forward(self,x,**kw):
        x=x.view(x.shape[0],-1)
        x=self.fc1(x)
        x=self.dropout(F.relu(x))
        x=self.fc2(x)
        x=F.relu(x)
        x=self.fc3(x)
        x=F.relu(x)
        x=self.fc4(x)
        x=F.relu(x)
        x=self.fc5(x)
        x=F.relu(x)
        h1=self.head1(x) #15 min
        h2=self.head2(x) #60 min
        h3=self.head3(x) #6 hrs
        return h1,h2,h3       

class SimpleFC5_block(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.fc1=nn.Linear(opt.n_max_seq*opt.d_model,opt.d_inner)
        self.dropout=nn.Dropout(opt.dropout)
        
        self.fc3_res=nn.Linear(opt.d_inner,opt.d_inner3)
        self.fc4_res=nn.Linear(opt.d_inner,opt.d_inner4)
        self.fc2_1=nn.Linear(opt.d_inner,opt.d_inner2)
        self.fc2_2=nn.Linear(opt.d_inner2,opt.d_inner2)
        self.fc3_1=nn.Linear(opt.d_inner2,opt.d_inner3)
        self.fc3_2=nn.Linear(opt.d_inner3,opt.d_inner3)
        self.fc3_3=nn.Linear(opt.d_inner3,opt.d_inner3)
        self.fc4_1=nn.Linear(opt.d_inner3,opt.d_inner4)
        self.fc4_2=nn.Linear(opt.d_inner4,opt.d_inner4)
        self.fc5=nn.Linear(opt.d_inner4,opt.d_inner5)
        self.head1=nn.Linear(opt.d_inner5,1)
        self.head2=nn.Linear(opt.d_inner5,1)
        self.head3=nn.Linear(opt.d_inner5,1)
    def forward(self,x,**kw):
        x=x.view(x.shape[0],-1)
        x=self.fc1(x)
        x=self.dropout(F.relu(x))
        res3=self.fc3_res(x)
        res4=self.fc4_res(x)
        x=self.fc2_1(x)
        x=F.relu(x)
        x=self.fc2_2(x)
        x=F.relu(x)
        x=self.fc3_1(x)
        x=F.relu(x)
        x=self.fc3_2(x)
        x=F.relu(x)
        x=self.fc3_3(x)
        x=F.relu(x)
        x=x+res3
        x=self.fc4_1(x)
        x=F.relu(x)
        x=self.fc4_2(x)
        x=F.relu(x)
        x=x+res4
        x=self.fc5(x)
        x=F.relu(x)
        h1=self.head1(x) #15 min
        h2=self.head2(x) #60 min
        h3=self.head3(x) #6 hrs
        return h1,h2,h3  

class LSTM(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.lstm=nn.LSTM(input_size = opt.d_model,
                          hidden_size = opt.d_inner,
                          batch_first = True,
                          dropout = opt.dropout,
                          num_layers = opt.n_layers,
                          bidirectional = opt.bidirect)
        self.head1=nn.Linear(opt.d_inner*opt.n_max_seq,1)
        self.head2=nn.Linear(opt.d_inner*opt.n_max_seq,1)
        self.head3=nn.Linear(opt.d_inner*opt.n_max_seq,1)
    def forward(self,x, **kw):
        #input batch x len x feature
        #output batch x len x d_inner(x 2 if bidirectional)
        output=self.lstm(x)
        a=output[0].contiguous().view(x.shape[0],-1)
        h1=self.head1(a) #15 min
        h2=self.head2(a) #60 min
        h3=self.head3(a) #6 hrs
        return h1,h2,h3

class GRU(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.gru=nn.GRU(input_size = opt.n_max_seq,
                          hidden_size = opt.d_inner,
                          batch_first = True,
                          dropout = opt.dropout,
                          num_layers = opt.n_layers,
                          bidirectional = opt.bidirect)
        self.head1=nn.Linear(opt.d_inner*opt.d_model,1)
        self.head2=nn.Linear(opt.d_inner*opt.d_model,1)
        self.head3=nn.Linear(opt.d_inner*opt.d_model,1)
    def forward(self,x, **kw):
        #input batch x len x feature
        #output batch x len x d_inner(x 2 if bidirectional)
        x=x.permute(0,2,1)
        output=self.gru(x)
        a=output[0].contiguous().view(x.shape[0],-1)
        h1=self.head1(a) #15 min
        h2=self.head2(a) #60 min
        h3=self.head3(a) #6 hrs
        return h1,h2,h3
    
class Transformer(nn.Module):
    '''
    multi head self-attention model
    '''
    def __init__(self,opt):
        super().__init__()
        #self.fc=nn.Linear(1, d_model)
        n_max_seq=opt.n_max_seq
        n_layers=opt.n_layers
        d_model=opt.d_model
        d_inner=opt.d_inner
        d_k=opt.d_k
        d_v=opt.d_v
        dropout=opt.dropout
        self.n_head=opt.n_head
        self.encoder_stack=nn.ModuleList([EncoderLayer(opt.n_head, d_model,d_inner,d_k,d_v, dropout) for _ in range(n_layers)])
#         self.conv1=nn.Conv1d(d_model,8,30,stride=3)
#         self.conv2=nn.Conv1d(d_model,8,30,stride=3)
#         self.conv3=nn.Conv1d(d_model,8,30,stride=3) # (n_max_seq-30)/stride-1
        
#         self.conv1_1=nn.Conv1d(8,16,30,stride=2)
#         self.conv2_1=nn.Conv1d(8,16,30,stride=2)
#         self.conv3_1=nn.Conv1d(8,16,30,stride=2) #
        
#         w=(n_max_seq-30)//3+1
#         w=(w-30)//2+1
#         self.head1=nn.Linear(w*16,1)
#         self.head2=nn.Linear(w*16,1)
#         self.head3=nn.Linear(w*16,1)
        self.head1=nn.Linear(n_max_seq*d_model,1)
        self.head2=nn.Linear(n_max_seq*d_model,1)
        self.head3=nn.Linear(n_max_seq*d_model,1)
    
    def forward(self, x, mask=None):
        #x=self.fc(x)
        b, len_q, d_model = x.size()
        #b=temporal_pos_emb+x + spatial_pos_emb#add pos emb to input
        if mask is not None:
            assert mask.dim() == 2
            assert mask.size(0) == x.size(0)
            mask=mask.repeat(1,len_q).view(b,len_q,len_q)
            mask=mask.repeat(self.n_head,1,1)
        
        for encoder in self.encoder_stack:
            a=encoder(x,mask=mask)
#         x=x.view(b,1,-1)
#         x1=self.conv1(x)
#         x1=self.conv1_1(x1)
#         x1=x1.view(b,-1)
#         h1=self.head1(x1)
        
#         x2=self.conv2(x)
#         x2=self.conv2_1(x2)
#         x2=x2.view(b,-1)
#         h2=self.head2(x2)
        
#         x3=self.conv3(x)
#         x3=self.conv3_1(x3)
#         x3=x3.view(b,-1)
#         h3=self.head3(x3)
        x=x.view(b,-1)
        h1=self.head1(x) #15 min
        h2=self.head2(x) #60 min
        h3=self.head3(x) #6 hrs

        return h1,h2,h3


class EncoderLayer(nn.Module):
    def __init__(self,  n_head,
                        d_model,
                        d_inner,
                        d_k,
                        d_v, 
                        dropout):
        super().__init__()
        self.selfatt=MultiHeadSelfAttention( n_head, d_model, d_k, d_v, dropout)
        self.layer_norm_selfatt=nn.LayerNorm(d_model)
        self.ffn=PointwiseFeedForward(d_model, d_inner, dropout)
        self.layer_norm_ffn=nn.LayerNorm(d_model)
    
    def forward(self, x, mask):
        resdential = x
        z=self.selfatt(x,mask=mask)
        z=z+resdential
        z=self.layer_norm_selfatt(z)
        output=self.ffn(z)
        output=output+z
        output=self.layer_norm_ffn(output)
        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout):
        super().__init__()
        self.n_head, self.d_k, self.d_v=n_head, d_k, d_v
        self.q_w=nn.Linear(d_model, d_k*n_head)
        self.k_w=nn.Linear(d_model, d_k*n_head)
        self.v_w=nn.Linear(d_model, d_v*n_head)
        nn.init.normal_(self.q_w.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.k_w.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.v_w.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.softmax=nn.Softmax(dim=2)
        self.dropout1=nn.Dropout(dropout)
        self.fc=nn.Linear(n_head*d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout2=nn.Dropout(dropout)

    def forward(self, x, k=None, v=None, mask=None):
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v
        b,len_q,_=x.size() #batch_size and max_length
        q=self.q_w(x).view(b, len_q, n_head, d_k)
        if k is not None:
            _, len_k, _ =k.size()
        else:
            k=self.k_w(x).view(b, len_q, n_head, d_k)
            len_k=len_q
        if v is None:
            v=self.v_w(x).view(b,len_q,n_head,d_v)
        q=q.permute(2,0,1,3).contiguous().view(-1,len_q,d_k) # need reduce to 3 dim as bmm only accept 3 dim
        k=k.permute(2,0,1,3).contiguous().view(-1,len_k,d_k)
        k_transpose=k.permute(0,2,1)
        v=v.permute(2,0,1,3).contiguous().view(-1,len_k,d_v)
        att=torch.bmm(q,k_transpose)
        att=att/np.sqrt(d_k)
        if mask is not None:
            att.masked_fill(mask, -np.inf) # mask have shape (Batch x len x len)
        att=self.softmax(att)
        rslt=self.dropout1(att)
        rslt=torch.bmm(att,v)
        rslt=rslt.view(n_head,b,len_k,d_v).permute(1,2,0,3).contiguous().view(b,len_k,n_head*d_v)
        rslt=self.fc(rslt)
        rslt=self.dropout2(rslt)
        return rslt
        
        
class PointwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.fc1=nn.Linear(d_model,d_inner)
        self.fc2=nn.Linear(d_inner,d_model)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        inner = F.relu(self.fc1(x))
        output = self.fc2(inner)
        output = self.dropout(output)
        return output

