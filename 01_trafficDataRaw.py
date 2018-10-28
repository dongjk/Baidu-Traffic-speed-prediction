import _pickle as pickle
import numpy as np
from collections import defaultdict

raw_speed_file = "./date/traffic_speed_sub-dataset"
gps_data_file = "./date/link_gps"

raw_data_file = "./clean/raw_speed_gps.pkl"

gps_data = {}
# raw_data = defaultdict(lambda : {'speed': np.zeros(5856)})
raw_data = {}

with open(gps_data_file, 'r') as f:
    for line in f:
        sp = line.split()
        id = sp[0].strip()
        gps1 = float(sp[1].strip())
        gps2 = float(sp[2].strip())
        gps = np.zeros(2)
        gps[0] = gps1
        gps[1] = gps2
        gps_data[id] = gps

not_have_gps = []

with open(raw_speed_file, 'r') as f:
    for line in f:
        sp = line.split(',')
        id = sp[0].strip()
        idx = int(sp[1].strip())
        speed = float(sp[2].strip())
        data_dict = {}
        if id not in raw_data:
            data_dict = {'speed': np.zeros(5856)}
            raw_data[id] = data_dict
        else:
            data_dict = raw_data[id]

        data_dict['speed'][idx] = speed

        if id not in gps_data:
            not_have_gps.append(id)
        else:
            data_dict['gps'] = gps_data[id]

print(len(raw_data))
print(len(not_have_gps))

with open(raw_data_file , 'wb') as f:
    pickle.dump(raw_data, file=f)
