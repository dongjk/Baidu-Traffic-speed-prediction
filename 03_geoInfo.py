import _pickle as pickle
from collections import defaultdict
import networkx as nx

import numpy as np

geo_data_raw = './date/road_network_sub-dataset'

link_id_dict_file = "./clean/link_id_dict.pkl"
geo_data = "./clean/geo.pkl"


geo_raw = {}


with open(link_id_dict_file , 'rb') as f:
    link_id_dict = pickle.load(f)

nodes_dict = defaultdict(lambda : {'in':set(), 'out':set()})
ids_set = set()
with open(geo_data_raw,'r') as f:
    for i, line in enumerate(f):
        if i ==0:
            continue
        # link_id	width	direction	snodeid	enodeid	length	speedclass	lanenum
        sp = line.split()
        id = sp[0].strip()
        width = int(sp[1].strip())
        direction = int(sp[2].strip())
        snodeid = sp[3].strip()
        enodeid = sp[4].strip()
        ids_set.add(id)
        s_links = nodes_dict[snodeid]
        e_links = nodes_dict[enodeid]

        if direction == 0 or direction == 1:

            s_links['in'].add(id)
            s_links['out'].add(id)

            e_links['in'].add(id)
            e_links['out'].add(id)
        elif direction == 2:
            s_links['out'].add(id)
            e_links['in'].add(id)
        elif direction == 3:
            s_links['in'].add(id)
            e_links['out'].add(id)
        else:
            raise Exception()

print(len(nodes_dict))
print(len(ids_set))
print(len(link_id_dict))

one_way_set = set()
link_dict = defaultdict(lambda : {'in':set(), 'out':set(), 'in_2':set(), 'out_2':set(), 'in_3':set(), 'out_3':set()})

for node, node_dict in nodes_dict.items():
    for in_link in node_dict['in']:
        for out_link in node_dict['out']:
            if in_link != out_link:
                one_way_set.add((in_link, out_link))
                link_dict[in_link]['out'].add(out_link)
                link_dict[out_link]['in'].add(in_link)

print(len(one_way_set))
print(len(link_dict))
# level2
for  node, node_dict in link_dict.items():
    in_set = node_dict['in']
    in_2_set =  node_dict['in_2']
    out_set = node_dict['out']
    out_2_set = node_dict['out_2']
    for in_node in in_set:
        in_set_1 = link_dict[in_node]['in']
        for in_node_1 in in_set_1:
            in_2_set.add((in_node_1, in_node))

    for out_node in out_set:
        out_set_1 = link_dict[out_node]['out']
        for out_node_1 in out_set_1:
            out_2_set.add((out_node, out_node_1))

# level3
for node, node_dict in link_dict.items():

    in_2_set = node_dict['in_2']

    out_2_set = node_dict['out_2']

    in_3_set = node_dict['in_3']

    out_3_set = node_dict['out_3']

    in_set = set()
    out_set = set()
    for in_node, _ in in_2_set:
        in_set.add(in_node)
    for _, out_node in out_2_set:
        out_set.add(out_node)

    for in_node in in_set:
        in_set_1 = link_dict[in_node]['in']
        for in_node_1 in in_set_1:
            in_3_set.add((in_node_1, in_node))

    for out_node in out_set:
        out_set_1 = link_dict[out_node]['out']
        for out_node_1 in out_set_1:
            out_3_set.add((out_node, out_node_1))

# other att:
with open(geo_data_raw,'r') as f:
    for i, line in enumerate(f):
        if i ==0:
            continue
        # link_id	width	direction	snodeid	enodeid	length	speedclass	lanenum
        sp = line.split()
        id = sp[0].strip()
        width = int(sp[1].strip())
        direction = int(sp[2].strip())
        length = float(sp[5].strip())
        speedclass = int(sp[6].strip())
        lanenum = int(sp[7].strip())
        # if id in link_dict:
        link_dict[id]['width'] = width
        link_dict[id]['direction'] = direction
        link_dict[id]['length'] = length
        link_dict[id]['speedclass'] = speedclass
        link_dict[id]['lanenum'] = lanenum

print(link_dict['1802838360064'])
print(link_dict['1802846360183'])
print(link_dict['1145870368871'])
print(link_dict['1452594443018'])


G = nx.DiGraph()
for in_link, out_link in one_way_set:
    speedclass = float(link_dict[out_link]['speedclass'])
    lanenum = float(link_dict[id]['lanenum'])
    G.add_edge(in_link, out_link, weight=lanenum/speedclass)

pagerank = nx.pagerank(G)

print(len(pagerank))
print(pagerank['1802838360064'])
print(pagerank['1802846360183'])
print(pagerank['1145870368871'])
print(pagerank['1452594443018'])

pagerak_list = np.zeros(len(pagerank))
for i, rank in enumerate(pagerank.values()):
    pagerak_list[i] = rank

mean = np.mean(pagerak_list)
std = np.std(pagerak_list)
print(mean)
print(std)

for node, node_dict in link_dict.items():
    rank = mean -1.96*std
    try:
        rank = pagerank[node]
    except Exception:
        pass
    rank_norm = (rank -mean)/std
    node_dict['rank'] = rank
    node_dict['rank_norm'] = rank_norm

in_len = []
out_len = []
in_2_len = []
out_2_len = []

in_3_len = []
out_3_len = []

rank_low = []

for node, node_dict in link_dict.items():
    in_len.append(len(node_dict['in']))
    if len(node_dict['in']) == 0:
        rank_low.append(node_dict['rank'])

    out_len.append(len(node_dict['out']))
    in_2_len.append(len(node_dict['in_2']))
    out_2_len.append(len(node_dict['out_2']))
    in_3_len.append(len(node_dict['in_3']))
    out_3_len.append(len(node_dict['out_3']))


check = []
check.append(in_len)
check.append(out_len)
check.append(in_2_len)
check.append(out_2_len)
check.append(in_3_len)
check.append(out_3_len)
check.append(rank_low)
for c in check:
    print("-------------------------------")
    print(max(c))
    print(min(c))
    print(np.mean(c))
    print(np.std(c))
    print("-------------------------------")

with open(geo_data , 'wb') as f:
    pickle.dump(dict(link_dict), file=f)

