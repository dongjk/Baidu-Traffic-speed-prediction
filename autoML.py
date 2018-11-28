import os
import random 
import traceback

def Transformer_auto():
    cmd="python train.py -model Transformer -dataset RoadDataSet3 -d_model=1 -cuda"
    epoch=random.randint(15,20)
    batch_size=random.choice(['32','64','128','256'])
    steps=random.choice(['640000','1280000','320000'])
    n_sample=random.randint(400,700)
    n_nb_sample=3
    n_layers=random.randint(1,2)
    n_max_seq=n_sample+28+n_nb_sample*44
    d_inner=random.choice(['256','512','1024','2048'])
    dropout=random.choice(['0','0.1','0.05'])
    cmd+=" -epoch="+str(epoch)
    cmd+=" -batch_size="+batch_size
    cmd+=" -steps="+steps
    cmd+=" -n_layers="+str(n_layers)
    cmd+=" -n_sample="+str(n_sample)
    cmd+=" -n_nb_sample="+str(n_nb_sample)
    cmd+=" -n_max_seq="+str(n_max_seq)
    cmd+=" -d_inner="+d_inner
    cmd+=" -dropout="+dropout
    return cmd


def SimpleFC_auto():
    cmd="python train.py -model SimpleFC -dataset RoadDataSet32 -d_model=1 -cuda"
    epoch=random.randint(40,60)
    batch_size=random.choice(['32','64','128','256'])
    steps=random.choice(['640000','1280000','320000'])
    n_sample=random.randint(400,700)
    n_nb_sample=random.randint(3,10)
    n_max_seq=n_sample+28+n_nb_sample*44
    d_inner=random.choice(['128','256','512','1024','2048','4096'])
    dropout=random.choice(['0','0','0.05'])
    cmd+=" -epoch="+str(epoch)
    cmd+=" -batch_size="+batch_size
    cmd+=" -steps="+steps
    cmd+=" -n_sample="+str(n_sample)
    cmd+=" -n_nb_sample="+str(n_nb_sample)
    cmd+=" -n_max_seq="+str(n_max_seq)
    cmd+=" -d_inner="+d_inner
    cmd+=" -dropout="+dropout
    return cmd

def SimpleFC2_auto():
    cmd="python train.py -model SimpleFC2 -dataset RoadDataSet32 -d_model=1 -cuda"
    epoch=random.randint(40,60)
    batch_size=random.choice(['32','64','128','256'])
    steps=random.choice(['640000','1280000','320000'])
    n_sample=random.randint(400,700)
    n_nb_sample=random.randint(3,10)
    n_max_seq=n_sample+28+n_nb_sample*44
    d_inner=random.choice(['128','256','512','1024','2048','4096'])
    d_inner2=random.choice(['128','256','512','1024','2048'])
    dropout=random.choice(['0','0','0.05'])
    cmd+=" -epoch="+str(epoch)
    cmd+=" -batch_size="+batch_size
    cmd+=" -steps="+steps
    cmd+=" -n_sample="+str(n_sample)
    cmd+=" -n_nb_sample="+str(n_nb_sample)
    cmd+=" -n_max_seq="+str(n_max_seq)
    cmd+=" -d_inner="+d_inner
    cmd+=" -d_inner2="+d_inner2
    cmd+=" -dropout="+dropout
    return cmd

def SimpleFC3_auto():
    cmd="python train.py -model SimpleFC3 -dataset RoadDataSet32 -d_model=1 -cuda"
    epoch=random.randint(30,40)
    batch_size=random.choice(['32','64','128','256'])
    steps=random.choice(['640000','1280000','320000'])
    n_sample=random.randint(400,1200)
    n_nb_sample=random.randint(8,40)
    n_max_seq=n_sample+28+n_nb_sample*44
    d_inner=random.choice(['256','512','1024','2048','4096'])
    d_inner2=random.choice(['256','512','1024','2048'])
    d_inner3=random.choice(['128','256','512','1024'])
    dropout=random.choice(['0','0','0.05'])
    cmd+=" -epoch="+str(epoch)
    cmd+=" -batch_size="+batch_size
    cmd+=" -steps="+steps
    cmd+=" -n_sample="+str(n_sample)
    cmd+=" -n_nb_sample="+str(n_nb_sample)
    cmd+=" -n_max_seq="+str(n_max_seq)
    cmd+=" -d_inner="+d_inner
    cmd+=" -d_inner2="+d_inner2
    cmd+=" -d_inner3="+d_inner3
    cmd+=" -dropout="+dropout
    return cmd

def SimpleFC4_auto():
    cmd="python train.py -model SimpleFC4 -dataset RoadDataSet32 -d_model=1 -cuda"
    epoch=random.randint(30,40)
    batch_size=random.choice(['32','64','128','256'])
    steps=random.choice(['640000','1280000','320000'])
    n_sample=random.randint(400,1200)
    n_nb_sample=random.randint(3,10)
    n_max_seq=n_sample+28+n_nb_sample*44
    d_inner=random.choice(['512','1024','2048','4096'])
    d_inner2=random.choice(['256','512','1024','2048'])
    d_inner3=random.choice(['128','256','512','1024'])
    d_inner4=random.choice(['128','256','512'])
    dropout=random.choice(['0','0','0.05'])
    cmd+=" -epoch="+str(epoch)
    cmd+=" -batch_size="+batch_size
    cmd+=" -steps="+steps
    cmd+=" -n_sample="+str(n_sample)
    cmd+=" -n_nb_sample="+str(n_nb_sample)
    cmd+=" -n_max_seq="+str(n_max_seq)
    cmd+=" -d_inner="+d_inner
    cmd+=" -d_inner2="+d_inner2
    cmd+=" -d_inner3="+d_inner3
    cmd+=" -d_inner4="+d_inner4
    cmd+=" -dropout="+dropout
    return cmd

def SimpleFC5_auto():
    cmd="python train.py -model SimpleFC5 -dataset RoadDataSet32 -d_model=1 -cuda"
    epoch=random.randint(30,40)
    batch_size=random.choice(['32','64'])
    steps=random.choice(['640000','1280000','320000'])
    n_sample=random.randint(400,800)
    n_nb_sample=random.randint(5,20)
    n_max_seq=n_sample+28+n_nb_sample*44
    d_inner=random.choice(['512','1024','2048'])
    d_inner2=random.choice(['256','512','1024'])
    d_inner3=random.choice(['128','256','512'])
    d_inner4=random.choice(['128','256'])
    d_inner5=random.choice(['128','256','512'])
    dropout=random.choice(['0','0','0.05'])
    cmd+=" -epoch="+str(epoch)
    cmd+=" -batch_size="+batch_size
    cmd+=" -steps="+steps
    cmd+=" -n_sample="+str(n_sample)
    cmd+=" -n_nb_sample="+str(n_nb_sample)
    cmd+=" -n_max_seq="+str(n_max_seq)
    cmd+=" -d_inner="+d_inner
    cmd+=" -d_inner2="+d_inner2
    cmd+=" -d_inner3="+d_inner3
    cmd+=" -d_inner4="+d_inner4
    cmd+=" -d_inner5="+d_inner5
    cmd+=" -dropout="+dropout
    return cmd

def SimpleFC5_block_auto():
    cmd="python train.py -model SimpleFC5_block -dataset RoadDataSet32 -d_model=1 -cuda"
    epoch=random.randint(30,40)
    batch_size=random.choice(['32','64','128','256'])
    steps=random.choice(['640000','1280000','320000'])
    n_sample=random.randint(800,1200)
    n_nb_sample=random.randint(10,30)
    n_max_seq=n_sample+29+n_nb_sample*44
    d_inner=random.choice(['512','1024'])
    d_inner2=random.choice(['512'])
    d_inner3=random.choice(['128','256'])
    d_inner4=random.choice(['128'])
    d_inner5=random.choice(['32','64'])
    dropout=random.choice(['0','0','0.05'])
    cmd+=" -epoch="+str(epoch)
    cmd+=" -batch_size="+batch_size
    cmd+=" -steps="+steps
    cmd+=" -n_sample="+str(n_sample)
    cmd+=" -n_nb_sample="+str(n_nb_sample)
    cmd+=" -n_max_seq="+str(n_max_seq)
    cmd+=" -d_inner="+d_inner
    cmd+=" -d_inner2="+d_inner2
    cmd+=" -d_inner3="+d_inner3
    cmd+=" -d_inner4="+d_inner4
    cmd+=" -d_inner5="+d_inner5
    cmd+=" -dropout="+dropout
    return cmd

def LSTM_auto():
    cmd="python train.py -model LSTM -dataset RoadDataSet4 -n_max_seq=45 -cuda"
    epoch=random.randint(10,20)
    batch_size=random.choice(['32','64','128','256','512'])
    steps=random.choice(['640000','1280000','320000'])
    n_layers=random.choice(['1','2','3'])
    n_sample=random.randint(8,120)
    d_model=n_sample+22
    d_inner=random.choice(['128','256','512','1024','2048','4096'])
    dropout=random.choice(['0','0','0.05'])
    cmd+=" -epoch="+str(epoch)
    cmd+=" -batch_size="+batch_size
    cmd+=" -steps="+steps
    cmd+=" -n_layers="+n_layers
    cmd+=" -n_sample="+str(n_sample)
    cmd+=" -d_model="+str(d_model)
    cmd+=" -d_inner="+d_inner
    cmd+=" -dropout="+dropout
    return cmd

def GRU_auto():
    cmd="python train.py -model GRU -dataset RoadDataSet4 -n_max_seq=45 -cuda"
    epoch=random.randint(10,20)
    batch_size=random.choice(['32','64','128','256','512'])
    steps=random.choice(['640000','1280000','320000'])
    n_layers=random.choice(['1','2','3'])
    n_sample=random.randint(8,120)
    d_model=n_sample+22
    d_inner=random.choice(['128','256','512','1024','2048','4096'])
    dropout=random.choice(['0','0','0.05'])
    cmd+=" -epoch="+str(epoch)
    cmd+=" -batch_size="+batch_size
    cmd+=" -steps="+steps
    cmd+=" -n_layers="+n_layers
    cmd+=" -n_sample="+str(n_sample)
    cmd+=" -d_model="+str(d_model)
    cmd+=" -d_inner="+d_inner
    cmd+=" -dropout="+dropout
    return cmd

def main():
    import sys
    automodel=eval(sys.argv[1]+'_auto')
    loops=int(sys.argv[2])
    for i in range(loops):
        cmd=automodel()
        print(cmd)
        os.system(cmd)

if __name__ == '__main__':
    main()
