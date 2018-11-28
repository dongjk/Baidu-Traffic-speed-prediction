
import argparse
import time
import numpy as np
from Layers import *
from tqdm import tqdm
import torch
import torch.optim as optim
import argparse
import time
import numpy as np
from Layers import *
from tqdm import tqdm

import torch.optim as optim
#this section provide test related functions

def cal_performance(a , b1,b2,b3):
    loss_15min = torch.sqrt(nn.MSELoss()(a[0],b1.unsqueeze(1)))
    loss_60min = torch.sqrt(nn.MSELoss()(a[1],b2.unsqueeze(1)))
    loss_6hour = torch.sqrt(nn.MSELoss()(a[2],b3.unsqueeze(1)))

    loss = loss_15min + loss_60min + loss_6hour

    acc_num_15min = get_acc_num(a[0],b1.unsqueeze(1),0.25) #get number of samples under 25% error.
    acc_num_60min = get_acc_num(a[1],b2.unsqueeze(1),0.25)
    acc_num_6hour = get_acc_num(a[2],b3.unsqueeze(1),0.25)

    return loss, loss_15min, loss_60min, loss_6hour, acc_num_15min, acc_num_60min, acc_num_6hour

def get_acc_num(pred, target, percent):
    p=pred.cpu().data.numpy()
    t=target.cpu().data.numpy()
    return (np.absolute(p-t)/t).mean()

def eval_epoch(model,val_data, optimizer, device, batch_size):
    ''' Epoch operation in val phase'''

    model.eval()

    total_loss=total_loss_15min=total_loss_60min=total_loss_6hour = 0
    total_acc_15min=total_acc_60min=total_acc_6hour = 0
    count = 0
    for batch in tqdm(
            val_data, mininterval=2,
            desc='  - (Testing)   ', leave=False):

        # prepare data
        src_seq,mask, t1,t2,t3 = map(lambda x: x.float().to(device), batch)
        # forward
#         optimizer.zero_grad()
        pred = model(src_seq,mask=mask.byte())

        # backward
        loss, loss_15min, loss_60min, loss_6hour, acc_num_15min, \
            acc_num_60min, acc_num_6hour = cal_performance(pred, t1,t2,t3)

        # note keeping
        total_loss += loss.item()
        total_loss_15min += loss_15min.item()
        total_loss_60min += loss_60min.item()
        total_loss_6hour += loss_6hour.item()
        total_acc_15min += acc_num_15min
        total_acc_60min += acc_num_60min
        total_acc_6hour += acc_num_6hour
        count += 1

    loss, loss_15min, loss_60min, loss_6hour = total_loss/count, \
        total_loss_15min/count, total_loss_60min/count, total_loss_6hour/count
    acc_15min, acc_60min, acc_6hour = 100*total_acc_15min/(count), \
        100*total_acc_60min/(count), 100*total_acc_6hour/(count)

    return loss, loss_15min, loss_60min, loss_6hour, acc_15min, acc_60min, acc_6hour
#use this function to load model, hyperparamaters.

def load_model(path):
    checkpoint = torch.load(path)
    #load hyper parameters
    opt=checkpoint['settings']
    #reconstruct model
    device = torch.device('cuda' if opt.cuda else 'cpu')
    model=ModelWrapper(opt.model,opt).model
    model = model.to(device)

    model=ModelWrapper(opt.model,opt).model
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total params:\t\t\t%d"%total_params)
    print("total trainable params:\t\t%d"%total_trainable_params)
    print('')
    print(opt)
    print('')
    model.load_state_dict(checkpoint['model'])
    opt.epoch=1
    if not hasattr(opt, 'n_nb_sample'):
        opt.n_nb_sample=3
    return opt, model

# load later half data and start testing.
import dataloader
from torch.utils.data import DataLoader
val_data_file = "./train_data/val_data2.pkl"

def start_test(path):
    opt, model = load_model(path)
    device = torch.device('cuda' if opt.cuda else 'cpu')

    ds=eval("dataloader."+opt.dataset)
    validation_data=DataLoader(ds(val_data_file,opt.n_sample,opt.steps,"test", n_nb_sample=opt.n_nb_sample),num_workers=1, batch_size=opt.batch_size)


    start = time.time()
    vloss, vloss_15min, vloss_60min, vloss_6hour, vacc_15min, vacc_60min, vacc_6hour =\
        eval_epoch(model, validation_data,None, device, opt.batch_size)
    print('  - (Testing) loss: {loss: 8.5f}, rmse_15min:{loss_15min:8.5f}, '
          'rmse_60min:{loss_60min:8.5f}, rmse_6hour:{loss_6hour:8.5f}, \n\t\t elapse: {elapse:3.1f} min,'\
          ' acc_15min: {acc_15min:3.3f} %, acc_60min: {acc_60min:3.3f} %,  acc_6hour: {acc_6hour:3.3f} %,'.format(
              loss=vloss, loss_15min=vloss_15min, loss_60min=vloss_60min, loss_6hour=vloss_6hour,
              acc_15min=vacc_15min, acc_60min=vacc_60min, acc_6hour=vacc_6hour, 
              elapse=(time.time()-start)/60))

def main():
    import sys
    path=sys.argv[1]+'/model_saved.chkpt'
    start_test(path)
if __name__ == '__main__':
    main()
