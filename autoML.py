import os
import random 
import traceback


def SimpleFC_auto():
    cmd="python train.py -model SimpleFC -dataset RoadDataSet3 -d_model=1 -cuda"
    epoch=random.randint(30,80)
    batch_size=random.choice(['32','64','128','256','512','1024','2048'])
    steps=random.choice(['640000','1280000','320000'])
    n_sample=random.randint(8,700)
    n_max_seq=n_sample+28+132
    d_inner=random.choice(['128','256','512','1024','2048','4096'])
    dropout=random.choice(['0','0','0.05'])
    cmd+=" -epoch="+str(epoch)
    cmd+=" -batch_size="+batch_size
    cmd+=" -steps="+steps
    cmd+=" -n_sample="+str(n_sample)
    cmd+=" -n_max_seq="+str(n_max_seq)
    cmd+=" -d_inner="+d_inner
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
    print(GRU_auto())
    for i in range(100):
        cmd=SimpleFC_auto()
        try:
            os.system(cmd)
        except Exception as e:
            print("type error: " + str(e))
            print(traceback.format_exc())
    for i in range(10):
        cmd=LSTM_auto()
        try:
            os.system(cmd)
        except Exception as e:
            print("type error: " + str(e))
            print(traceback.format_exc())
    for i in range(10):
        cmd=GRU_auto()
        try:
            os.system(cmd)
        except Exception as e:
            print("type error: " + str(e))
            print(traceback.format_exc())

if __name__ == '__main__':
    main()
