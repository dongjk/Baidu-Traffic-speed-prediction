# Baidu Traffic speed prediction

This repo is implementation of "[Baidu Traffic Speed Prediction](https://ai.baidu.com/broad/subordinate?dataset=traffic)"

A random mechanism is used in this implementation, program will choose Model from **self-attention**, **GRU**, **1-5 hidden layers Dense net**, also random choose **hyper parameters** and random choose **data features** to train.

Further idea is using nerual arhitecture search to do model evolution.

# Requirement
- python 3.4+
- pytorch 0.4.1+
- tqdm
- numpy


# Usage

### 0) Download the data.

Download data in this [link]( https://ai.baidu.com/broad/download?dataset=traffic)

In this notebook only use `traffic_speed_sub-dataset.zip` and `road_network_sub-dataset.zip` packages.

### 1) Preprocess the data.
Create data folder, and unzip upper two packages to it
```bash
mkdir data
unzip traffic_speed_sub-dataset.zip -d ./data
unzip road_network_sub-dataset.zip -d ./data
```
create clean folder
```bash
mkdir clean
```

run data processing scripts

```bash
python 01_trafficDataRaw.py
python 02_extract_link_id_dict.py
python 03_geoInfo.py
python 04_timeInformation.py
python 05_perpare_train_data.py
```
dataset is splited into first half and second half, this notebook will only use first half to train and validation.

### 2) Randomly choose models and features and train
Example:
```bash
python autoML.py SimpleFC 1      # choose SimpleFC model and try once.
python autoML.py SimpleFC3 5     # choose SimpleFC3 model and try 5 times.
python autoML.py Transformer 1    # choose Transfomer model and try once.
python autoML.py GRU 1           # choose GRU model and try once.
```
> currently support model including: SimpleFC(1 hidden layer Dense net), SimpleFC2(2 hidden layer Dense net) ... to SimpleFC5(5 hidden layer Dense net), SimpleFC5_block(residential 9 hidden layer Dense net.), Transformer(multi head self-attention), GRU.


### 3) View result
step 2 will generate one folder for each try, run plot_result.ipynb to view results.


