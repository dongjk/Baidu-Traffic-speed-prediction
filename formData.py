import _pickle as pickle
import numpy as np



update_file = './X.p'

norm_file = './norm.p'


with open(update_file, 'rb') as f:

    raw_data = pickle.load(f)



norm_data[:,:,3] = np.log(norm_data[:,:,3] + 1e-7)

norm_data[:,:,5:8] = np.log(norm_data[:,:,5:8] + 1e-7)




mean = np.mean(norm_data, axis=0)

std = np.std(norm_data, axis=0) + 1e-8


norm_data = (norm_data - mean) /std

max_data = mean +1.96*std

min_data = mean - 1.96*std
norm_data = np.maximum(min_data, norm_data)

norm_data = np.minimum(norm_data, max_data)

for i in range(72):
    norm_data[:,i,8] = (i%24 - 11/5)/6.922186552431729
    print(norm_data[20,i,:])


all_norm = {'std':std, 'mean':mean, 'data':norm_data}

with open(norm_file, 'wb') as f:
    pickle.dump(all_norm, f)

test = np.zeros(24)

for i in range(24):
    test[i] = i
print(np.mean(test))
print(np.std(test))