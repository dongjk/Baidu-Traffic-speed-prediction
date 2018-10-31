import argparse
import time
import numpy as np
from Layers import *
from tqdm import tqdm

import torch.optim as optim

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
    return (np.absolute(p-t)/t < percent).sum()

def train_epoch(model, training_data, optimizer, device, batch_size):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss=total_loss_15min=total_loss_60min=total_loss_6hour = 0
    total_acc_15min=total_acc_60min=total_acc_6hour = 0
    count = 0
    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        src_seq,mask, t1,t2,t3 = map(lambda x: x.float().to(device), batch)
        b_size=src_seq.shape[0]
        # forward
        optimizer.zero_grad()
        pred = model(src_seq,mask=mask.byte())

        # backward
        loss, loss_15min, loss_60min, loss_6hour, acc_num_15min, \
            acc_num_60min, acc_num_6hour = cal_performance(pred, t1,t2,t3)
        loss_15min.backward(retain_graph=True)
        loss_60min.backward(retain_graph=True)
        loss_6hour.backward()

        # update parameters
        optimizer.step()

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
    acc_15min, acc_60min, acc_6hour = 100*total_acc_15min/(count*batch_size), \
        100*total_acc_60min/(count*batch_size), 100*total_acc_6hour/(count*batch_size)
    
    return loss, loss_15min, loss_60min, loss_6hour, acc_15min, acc_60min, acc_6hour

def eval_epoch(model,val_data, optimizer, device, batch_size):
    ''' Epoch operation in val phase'''

    model.eval()

    total_loss=total_loss_15min=total_loss_60min=total_loss_6hour = 0
    total_acc_15min=total_acc_60min=total_acc_6hour = 0
    count = 0
    for batch in tqdm(
            val_data, mininterval=2,
            desc='  - (Validation)   ', leave=False):

        # prepare data
        src_seq,mask, t1,t2,t3 = map(lambda x: x.float().to(device), batch)
        # forward
        optimizer.zero_grad()
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
    acc_15min, acc_60min, acc_6hour = 100*total_acc_15min/(count*batch_size), \
        100*total_acc_60min/(count*batch_size), 100*total_acc_6hour/(count*batch_size)
    
    return loss, loss_15min, loss_60min, loss_6hour, acc_15min, acc_60min, acc_6hour


def train(model, training_data, validation_data, optimizer, device, opt):
    log_train_file = opt.logdir + '/train.log'
    log_valid_file = opt.logdir + '/valid.log'
    
    valid_accus = []
    rmses_15min = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        loss, loss_15min, loss_60min, loss_6hour, acc_15min, acc_60min, acc_6hour =\
            train_epoch(model, training_data, optimizer, device, opt.batch_size)
        print('  - (Training)   loss: {loss: 8.5f}, rmse_15min:{loss_15min:8.5f}, '
              'rmse_60min:{loss_60min:8.5f}, rmse_6hour:{loss_6hour:8.5f}, \n\t\t elapse: {elapse:3.1f} min,'\
              ' acc_15min: {acc_15min:3.3f} %, acc_60min: {acc_60min:3.3f} %,  acc_6hour: {acc_6hour:3.3f} %,'.format(
                  loss=loss, loss_15min=loss_15min, loss_60min=loss_60min, loss_6hour=loss_6hour,
                  acc_15min=acc_15min, acc_60min=acc_60min, acc_6hour=acc_6hour, 
                  elapse=(time.time()-start)/60))

        start = time.time()
        vloss, vloss_15min, vloss_60min, vloss_6hour, vacc_15min, vacc_60min, vacc_6hour =\
            eval_epoch(model, validation_data,optimizer, device, opt.batch_size)
        print('  - (Validation) loss: {loss: 8.5f}, rmse_15min:{loss_15min:8.5f}, '
              'rmse_60min:{loss_60min:8.5f}, rmse_6hour:{loss_6hour:8.5f}, \n\t\t elapse: {elapse:3.1f} min,'\
              ' acc_15min: {acc_15min:3.3f} %, acc_60min: {acc_60min:3.3f} %,  acc_6hour: {acc_6hour:3.3f} %,'.format(
                  loss=vloss, loss_15min=vloss_15min, loss_60min=vloss_60min, loss_6hour=vloss_6hour,
                  acc_15min=vacc_15min, acc_60min=vacc_60min, acc_6hour=vacc_6hour, 
                  elapse=(time.time()-start)/60))

        valid_accus += [vloss]
        rmses_15min += [vloss_15min]
        if vloss_15min<3.71:
             with open(opt.logdir + '/newrecord', 'a') as f:
                f.write('{loss_15min:8.5f}\n'.format(loss_15min=vloss_15min))
        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.logdir+'/'+opt.save_model + '.chkpt'
                if vloss <= min(valid_accus) or vloss_15min <= min(rmses_15min):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{loss_15min:8.5f},{loss_60min:8.5f},{loss_6hour:8.5f},\
                          {acc_15min:3.3f},{acc_60min:3.3f},{acc_6hour:3.3f}\n'.format(
                epoch=epoch_i, loss=loss, loss_15min=loss_15min, loss_60min=loss_60min, loss_6hour=loss_6hour,
                  acc_15min=acc_15min, acc_60min=acc_60min, acc_6hour=acc_6hour))
            log_vf.write('{epoch},{loss: 8.5f},{loss_15min:8.5f},{loss_60min:8.5f},{loss_6hour:8.5f},\
                          {acc_15min:3.3f},{acc_60min:3.3f},{acc_6hour:3.3f}\n'.format(
                epoch=epoch_i, loss=vloss, loss_15min=vloss_15min, loss_60min=vloss_60min, loss_6hour=vloss_6hour,
                  acc_15min=vacc_15min, acc_60min=vacc_60min, acc_6hour=vacc_6hour))

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-model', default="Transformer")
    parser.add_argument('-dataset', default="RoadDataSet")
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=1024)
    parser.add_argument('-steps', type=int, default=5000)

    parser.add_argument('-n_max_seq', type=int, default=30)
    parser.add_argument('-n_sample', type=int, default=24)
    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_inner', type=int, default=256)
    parser.add_argument('-d_k', type=int, default=8)
    parser.add_argument('-d_v', type=int, default=8)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0)
    parser.add_argument('-bidirect', type=bool, default=False)

    parser.add_argument('-logdir', default='./')
    parser.add_argument('-save_model', default="model_saved")
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-cuda', action='store_true')

    opt = parser.parse_args()


    device = torch.device('cuda' if opt.cuda else 'cpu')

#     n_sample=opt.n_sample+1+44*3
#    n_max_seq=4+2+opt.n_sample+1+44*3
    model=ModelWrapper(opt.model,opt).model
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total params:\t\t\t%d"%total_params)
    print("total trainable params:\t\t%d"%total_trainable_params)
    print('')
    print(opt)
    print('')
    
    #create folder for this training run
    import datetime,os
    now = datetime.datetime.now()
    newDirName = now.strftime("%Y%m%d-%H%M%S")
    opt.logdir=opt.logdir+newDirName
    os.mkdir(opt.logdir)
    print('Log folder: %s' % opt.logdir)
    log_hyp_file = opt.logdir + '/hyperpara.txt'
    with  open(log_hyp_file, 'a') as f:
        f.write('%s' % opt)

    import dataloader
    from torch.utils.data import DataLoader
    train_data_file = "./train_data/train_data2.pkl"
    val_data_file = "./train_data/val_data2.pkl"
    
    ds=eval("dataloader."+opt.dataset)
    training_data=DataLoader(ds(train_data_file,opt.n_sample,opt.steps,"train"), batch_size=opt.batch_size)
    validation_data=DataLoader(ds(train_data_file,opt.n_sample,opt.steps,"val"), batch_size=opt.batch_size)

    optimizer= optim.Adam(model.parameters())
    train(model, training_data, validation_data, optimizer, device ,opt)
if __name__ == '__main__':
    main()
