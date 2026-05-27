import numpy as np
import networkx as nx
import hashlib

def get_random_walks(G, num_walks, walk_length):
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                cur = walk[-1]
                cur_nbrs = list(G.neighbors(cur))
                if len(cur_nbrs) > 0:
                    walk.append(np.random.choice(cur_nbrs))
                else:
                    break
            walks.append([str(n) for n in walk])
    return walks

def simhash(label_list):
    # Simplified SimHash for label correlations
    if not label_list:
        return "0"
    v = np.zeros(64)
    for label in label_list:
        h = bin(int(hashlib.md5(str(label).encode()).hexdigest(), 16))[-64:].zfill(64)
        for i in range(64):
            if h[i] == '1':
                v[i] += 1
            else:
                v[i] -= 1
    fingerprint = ""
    for i in range(64):
        if v[i] >= 0:
            fingerprint += "1"
        else:
            fingerprint += "0"
    return str(int(fingerprint, 2))