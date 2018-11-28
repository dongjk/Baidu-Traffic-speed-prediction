import dataloader
from torch.utils.data import DataLoader
train_data_file = "./train_data/train_data2.pkl"
val_data_file = "./train_data/val_data2.pkl"

ds=dataloader.RoadDataSet3
#training_data=DataLoader(ds(train_data_file,8000,640,"train", n_nb_sample=5),num_workers=1,  batch_size=64)
d=ds(train_data_file,8000,640,"train", n_nb_sample=5)
d.__getitem__(0)
