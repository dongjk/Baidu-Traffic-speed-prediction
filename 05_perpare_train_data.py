import _pickle as pickle
import numpy as np

TRAIN_LENGTH = 30*24*4

raw_data_file = "./clean/raw_speed_gps.pkl"
link_id_dict_file = "./clean/link_id_dict.pkl"

output_time_feature_file = "./clean/time_feature_15min.pkl"
geo_data_file = "./clean/geo.pkl"


train_data_file = "./train_data/train_data2.pkl"

val_data_file = "./train_data/val_data2.pkl"

id_idx_gps_file = "./train_data/id_idx_gps.txt"

train_one_month = {}
val_one_month = {}


with open(raw_data_file , 'rb') as f:
    raw = pickle.load(f)

with open(output_time_feature_file, 'rb') as f:
    time_feature = pickle.load(f)


with open(geo_data_file, 'rb') as f:
    geo_data = pickle.load(f)


with open(link_id_dict_file, 'rb') as f:
    link_id_dict = pickle.load(f)

id_link_dict = {}
for link_id, idx in link_id_dict.items():
    id_link_dict[idx] = link_id

#last one is for zero_padding link for CNN
link_attrs = np.zeros((len(id_link_dict) +1, 21))

link_idx_gps = []

for id in range(len(id_link_dict)):
    link_id = id_link_dict[id]
    raw_dict = raw[link_id]
    geo_info = geo_data[link_id]
    attrs  = np.zeros(21)

    width = geo_info['width']
    width_onehot = np.zeros(4)
    # 15 0 30 1 55 2 130 3
    width_onehot[min(width //20, 3)] = 1

    direction = geo_info['direction']
    direction_onehot = np.zeros(4)
    direction_onehot[direction] =1

    length =  geo_info['length']

    speedclass = geo_info['speedclass']
    speedclass_onehot = np.zeros(8)
    speedclass_onehot[speedclass - 1] = 1

    lanenum = geo_info['lanenum']
    lanenum_onehot = np.zeros(3)
    lanenum_onehot[lanenum -1] = 1

    pagerank = geo_info['rank']

    attrs[0:4] = width_onehot
    attrs[4:8] = direction_onehot
    attrs[8:16] = speedclass_onehot
    attrs[16:19] = lanenum_onehot
    attrs[19] = length
    attrs[20] = pagerank

    link_attrs[id, :] = attrs

    gps0 = raw_dict['gps'][0]
    gps1 = raw_dict['gps'][1]
    link_idx_gps.append((link_id, id, gps0, gps1))


link_idx_gps.append(("-1", len(id_link_dict), 0, 0))
link_idx_gps=np.asarray(link_idx_gps).astype(np.float)
mean = np.mean(link_idx_gps[:, 2], axis=0)
std = np.std(link_attrs[:, 2], axis=0)
gpsa=(link_idx_gps[:, 2]-mean)/std
mean = np.mean(link_idx_gps[:, 3], axis=0)
std = np.std(link_attrs[:, 3], axis=0)
gpsb=(link_idx_gps[:, 3]-mean)/std
gps=np.concatenate((gpsa.reshape((-1,1)),gpsb.reshape((-1,1))),axis=1)

with open(id_idx_gps_file, 'w') as f:
    for link_id, id, gps0, gps1 in link_idx_gps:

        print(str(link_id) +" " + str(id) + " " +str(gps0) + " " +str(gps1), file=f)


#let's do norm for length and rank

mean = np.mean(link_attrs[:, 19:21], axis=0)
std = np.std(link_attrs[:, 19:21], axis=0)
print(mean)
print(std)
link_attrs[:, 19:21] = (link_attrs[:, 19:21] - mean)/std


train_one_month['link_attrs'] = link_attrs

val_one_month['link_attrs'] = link_attrs

train_one_month['mean_len_rank'] = mean
train_one_month['std_len_rank'] = std

val_one_month['mean_len_rank'] = mean
val_one_month['std_len_rank'] = std


#last one is for zero_padding link for CNN
full_raw = np.zeros((len(id_link_dict) +1, 5856))

for id in range(len(id_link_dict)):
    link_id = id_link_dict[id]
    full_raw[id, :] = raw[link_id]['speed']

# we have to predict 15 30 1h 6h, last 24H data used for perdict only, first 24 data used for input only
train_raw = full_raw[:, 0:TRAIN_LENGTH +24]
val_raw = full_raw[:, TRAIN_LENGTH -24:]

#let's norm speed
mean_speed = np.mean(train_raw)
std_speed = np.std(train_raw)

#train_raw = (train_raw - mean_speed)/std_speed
#val_raw = (val_raw - mean_speed)/std_speed

train_one_month['speed_ary'] = train_raw
val_one_month['speed_ary'] = val_raw

train_one_month['mean_speed_train'] = mean_speed
train_one_month['std_speed_train'] = std_speed

val_one_month['mean_speed_train'] = mean_speed
val_one_month['std_speed_train'] = std_speed

train_one_month['gps'] = gps
val_one_month['gps'] = gps

print(mean_speed)
print(std_speed)


# we have to predict 15 30 1h 6h, last 24H data used for perdict only, first 24 data used for input only
train_time_feature = time_feature[0:TRAIN_LENGTH+24, :]
val_time_feature = time_feature[TRAIN_LENGTH-24:, :]

train_one_month['time_feature'] = train_time_feature
val_one_month['time_feature'] = val_time_feature


#lets get idx for level 1 level2 level3 in/out link for each data.
#assume that based on (mean +1.96*std)
# 3 top rank for level 1
# 7 top rank for level 2
# 12 top rank for level 3
# use PADDING link as mask.
# total size 44 * 21  array.

geo_nebor_attrs = np.zeros((len(id_link_dict), 44, 21))
geo_nebor_idx = np.zeros((len(id_link_dict) ,44), dtype=np.int32)

# default is padding link idx
geo_nebor_idx += len(id_link_dict)


def update_geo_info(idx, start_idx, max_cout, links):
    link_list = list(links)
    link_list.sort(key = lambda x : geo_data[x]['rank'], reverse=True)

    for i in range(start_idx, start_idx+min(len(link_list),max_cout)):
        id_neb = link_id_dict[link_list[i - start_idx ]]
        geo_nebor_idx[idx, i] = id_neb
        geo_nebor_attrs[idx, i, :] = link_attrs[id_neb]




for idx in range(len(id_link_dict)):
    link_id = id_link_dict[idx]
    node_dict = geo_data[link_id]

    in_2_set = node_dict['in_2']
    out_2_set = node_dict['out_2']

    in_3_set = node_dict['in_3']
    out_3_set = node_dict['out_3']

    in_set = node_dict['in']
    out_set = node_dict['out']

    update_geo_info(idx, 0, 3, in_set)
    update_geo_info(idx, 22, 3, out_set)

    update_geo_info(idx, 3, 7, [x[0] for x in in_2_set])
    update_geo_info(idx, 25, 7, [x[1] for x in out_2_set])

    update_geo_info(idx, 10, 12, [x[0] for x in in_3_set])
    update_geo_info(idx, 32, 12, [x[1] for x in out_3_set])



train_one_month['geo_nebor_attrs'] = geo_nebor_attrs
val_one_month['geo_nebor_attrs'] = geo_nebor_attrs


train_one_month['geo_nebor_idx'] = geo_nebor_idx
val_one_month['geo_nebor_idx'] = geo_nebor_idx

mask=np.zeros(geo_nebor_idx.shape)
mask[geo_nebor_idx==44172]=1
train_one_month['mask'] = mask
val_one_month['mask'] = mask

with open(train_data_file, 'wb') as f:
    pickle.dump(train_one_month, file=f)

with open(val_data_file, 'wb') as f:
    pickle.dump(val_one_month, file=f)

