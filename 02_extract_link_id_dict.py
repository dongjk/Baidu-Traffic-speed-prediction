import _pickle as pickle

raw_data_file = "./clean/raw_speed_gps.pkl"
link_id_dict_file = "./clean/link_id_dict.pkl"


with open(raw_data_file , 'rb') as f:
    raw = pickle.load(f)


keys = {}
for i, k in enumerate(raw.keys()):
    keys[k] = i


print(len(keys))
print(len(raw))
with open(link_id_dict_file , 'wb') as f:
    pickle.dump(keys, file=f)


