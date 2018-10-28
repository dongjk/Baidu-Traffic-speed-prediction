import os
import math
from collections import defaultdict
import _pickle as pickle
import numpy as np

"""
    time feature extraction
"""

d_pe=8
TIME_FEATURE_DIM = 4 +d_pe*2
output_time_feature_file = "./clean/time_feature_15min.pkl"

"""
Monday - Sunday
day = [0,  1,
           2,  3,  4,  5,  6,  7,  8,
           9,  10,  11,  12,  13,  14,  15,
           16,  17,  18,  19,  20,  21,  22,
           23,  24,  25,  26,  27,  28,  29,
           30,  31,  32,  33,  34,  35,  36,
           37,  38,  39,  40,  41,  42,  43,
           44,  45,  46,  47,  48,  49,  50,
           51,  52,  53,  54,  55,  56,  57,
           58,  59,  60
]
"""
workday = [0,
           4,  5,  6,
           9,  10,  11,  12,  13,
           16,  17,  18,  19,  20,
           23,  24,  25,  26,  27,
           31,  32,  33,  34,
           37,  38,  39,  40,  41,
           44,  45,  46,  47,  48,
           51,  52,  53,  54,  55,  56,
           60
           ]
weekend = [
    7, 8,
    14, 15,
    21, 22,
    35, 36,
    42, 43,
    49, 50,
]
festival = [
    1, 2, 3,  # Tomb Sweeping Day
    28, 29, 30,  # May Day
    57, 58, 59  # Dragon Boat Festival
]
"""
    ### extract time feature ###
    period: 1 Apr, 2017 - 31 May, 2017
    workday, holiday(weekend or festival): one-hot, 3 dim
    hour: float, 1 dim
    min: float, 1 dim
    peak hour(7:00-10:00, 17:00-20:00): float, 1 dim
    time_feature_dim = 6
"""


def time_feature_extraction(time):
    temp_time_feature = np.zeros(TIME_FEATURE_DIM, dtype=np.float)
    day = int(time / (24 * 4))
    if day in workday:
        temp_time_feature[0] = 1
    elif day in weekend:
        temp_time_feature[1] = 1
    else:
        temp_time_feature[2] = 1
    hour = int((time - day * 24 * 4) / 4)
    min = time - day * 24 * 4 - hour * 4

    if (hour in range(6, 9)) or (hour in range(16, 19)):
        temp_time_feature[3] = 1
    return temp_time_feature

def position_encoding_init(n_position, emb_dim):
    ''' Init the sinusoid position encoding table '''

    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)] for pos in range(n_position)])

    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

    return position_enc

one_day_endcode = position_encoding_init(24*4, d_pe)

one_week_endcode = position_encoding_init(24*4*7, d_pe)

TOTAL_TIME = 61 * 24 * 4  # 61 days * 24 hours * 4 mins(15 min interval)
time_feature = np.zeros((TOTAL_TIME, TIME_FEATURE_DIM), dtype=np.float)
for time in range(TOTAL_TIME):
    time_feature[time, :] = time_feature_extraction(time)
    day_pos = time % (24*4)
    week_pos = time % (24 * 4 * 7)
    time_feature[time, 4:4+d_pe] = one_day_endcode[day_pos]
    time_feature[time, 4+d_pe:] = one_week_endcode[week_pos]


print(time_feature[0])
print(time_feature[24*4])
print(time_feature[24*4*7])

print(time_feature[1])
print(time_feature[24*4 +1] )
print(time_feature[24*4*7 +1])

pickle.dump(time_feature, open(output_time_feature_file, "wb"))
