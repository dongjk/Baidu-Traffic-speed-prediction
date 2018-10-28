import argparse
import time
from Layers import *
from tqdm import tqdm

import torch.optim as optim

def cal_performance(a , b1,b2,b3):
    p1= nn.MSELoss()(a[0],b1.unsqueeze(1))
    p2= nn.MSELoss()(a[1],b2.unsqueeze(1))
    p3= nn.MSELoss()(a[2],b3.unsqueeze(1))
    return torch.sqrt(p1*.6 + p2*.0 + p3*.0)

def train_epoch(model, training_data, optimizer, device):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        src_seq,mask, t1,t2,t3 = map(lambda x: x.float().to(device), batch)
        b_size=src_seq.shape[0]
        # forward
        optimizer.zero_grad()
        pred = model(src_seq,mask.byte())

        # backward
        loss = cal_performance(pred, t1,t2,t3)
        loss.backward()

        # update parameters
        optimizer.step()

        # note keeping
        total_loss += loss.item()

    loss_per_word = total_loss / b_size
    return loss_per_word

def eval_epoch(model,val_data, optimizer, device):
    ''' Epoch operation in val phase'''

    model.eval()

    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(
                val_data, mininterval=2,
                desc='  - (Validation)   ', leave=False):

            # prepare data
            src_seq, mask,t1,t2,t3 = map(lambda x: x.to(device).float(), batch)
            b_size=src_seq.shape[0]
            # forward
            optimizer.zero_grad()
            pred = model(src_seq,mask.byte())

            # backward
            loss = cal_performance(pred, t1,t2,t3)

            # note keeping
            total_loss += loss.item()

    loss_per_word = total_loss/b_size
    return loss_per_word 

def train(model, training_data, validation_data, optimizer, device, opt):
    log_train_file = opt.logdir + '/train.log'
    log_valid_file = opt.logdir + '/valid.log'

    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss = train_epoch(model, training_data, optimizer, device)
        print('  - (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  loss=train_loss, accu=100,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss = eval_epoch(model, validation_data,optimizer, device)
        print('  - (Validation) loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    loss=valid_loss,  accu=1,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_loss]

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
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f}\n'.format(
                epoch=epoch_i, loss=train_loss
                ))
            log_vf.write('{epoch},{loss: 8.5f}\n'.format(
                epoch=epoch_i, loss=valid_loss
                ))

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=1024)

    parser.add_argument('-n_sample', type=int, default=24)
    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_inner', type=int, default=256)
    parser.add_argument('-d_k', type=int, default=8)
    parser.add_argument('-d_v', type=int, default=8)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-logdir', default='./')
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-cuda', action='store_true')

    opt = parser.parse_args()


    device = torch.device('cuda' if opt.cuda else 'cpu')

    n_sample=opt.n_sample+1+44*3
    model = Transformer(n_head=opt.n_head,
                                n_sample=n_sample,
                                n_layers=opt.n_layers,
                                d_model=opt.d_model,
                                d_inner=opt.d_inner,
                                d_k=opt.d_k,
                                d_v=opt.d_v,
                                dropout=opt.dropout).to(device)
    

    #create folder for this training run
    import datetime,os
    now = datetime.datetime.now()
    newDirName = now.strftime("%Y%m%d-%H%M%S")
    opt.logdir=opt.logdir+newDirName
    os.mkdir(opt.logdir)

    from dataloader import RoadDataSet2
    from torch.utils.data import DataLoader
    

    train_data_file = "./train_data/train_data.pkl"
    val_data_file = "./train_data/val_data.pkl"

    training_data=DataLoader(RoadDataSet2(train_data_file,opt.n_sample,20), batch_size=opt.batch_size)
    validation_data=DataLoader(RoadDataSet2(val_data_file,opt.n_sample,4), batch_size=opt.batch_size)

    optimizer= optim.Adam(model.parameters())
    train(model, training_data, validation_data, optimizer, device ,opt)
if __name__ == '__main__':
    main()
