import torch
import torch.nn as nn
import numpy as np

class Transformer(nn.Module):
    def __init__(self,n_sample,
                    n_head,
                    n_layers,
                    d_model,
                    d_inner,
                    d_k,
                    d_v,
                    dropout=0.1):
        super().__init__()
        self.fc=nn.Linear(1, d_model)
        self.encoder_stack=nn.ModuleList([EncoderLayer(n_head, d_model,d_inner,d_k,d_v, dropout)] * n_layers)
        self.head1=nn.Linear(n_sample*d_model,1)
        self.head2=nn.Linear(n_sample*d_model,1)
        self.head3=nn.Linear(n_sample*d_model,1)
    
    def forward(self, x, mask=None):
        x=self.fc(x)
        b, len_q, d_model = x.size()
        #b=temporal_pos_emb+x + spatial_pos_emb#add pos emb to input
        if mask is not None:
            assert mask.dim() == 2
            assert mask.size(0) == x.size(0)
            mask=mask.repeat(1,len_q).view(b,len_q,len_q)
            mask=mask.repeat(8,1,1)
        
        for encoder in self.encoder_stack:
            a=encoder(x,mask=mask)
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
            att=att.masked_fill(mask, -np.inf) # mask have shape (Batch x len x len)
        att=self.softmax(att)
        rslt=self.dropout1(att)
        rslt=torch.bmm(att,v)
        rslt=rslt.view(n_head,b,len_k,d_v).permute(1,2,0,3).contiguous().view(b,len_k,n_head*d_v)
        rslt=self.dropout2(rslt)
        return rslt
        
        
class PointwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.fc1=nn.Linear(d_model,d_inner)
        self.fc2=nn.Linear(d_inner,d_model)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        inner = self.fc1(x)
        output = self.fc2(inner)
        output = self.dropout(output)
        return output

