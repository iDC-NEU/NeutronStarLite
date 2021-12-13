from __future__ import print_function
import numpy as np
import random
import json
import sys
import os
import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
print (major)
print (minor)
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

def load_data(normalize=True, load_walks=False, inter=""):
    prefix= "reddit"

    G_data = json.load(open(prefix + "-G"+inter+".json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}
    
    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    mask_file= open(prefix+'.mask',mode='w')
    for node in G.nodes():
        if G.node[node]['val']:
            mask_file.write(str(id_map[node])+" val\n")
        elif G.node[node]['test']:
            mask_file.write(str(id_map[node])+" test\n")
        else:
            mask_file.write(str(id_map[node])+" train\n")
    mask_file.close()
    label_file=open(prefix+'.labeltable',mode='w')
    for node in G.nodes():
        label_file.write(str(id_map[node])+" "+str(class_map[node])+"\n")
    label_file.close()
    print ('shape',feats.shape)
    np.savetxt(prefix+'.featuretable',feats,fmt='%.18f',delimiter=' ')
    #feature_file=open('reddit.featuretable',mode='w')
    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    np.savetxt(prefix+'.featuretablenorm',feats,fmt='%.18f',delimiter=' ')
    edge_file=open(prefix+'.edge.txt',mode='w')
    for edge in G.edges():
       edge_file.write(str(id_map[edge[0]])+" "+str(id_map[edge[1]])+"\n")
    edge_file.close()

    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False
    return G, feats, id_map, walks, class_map

if __name__ == "__main__":
    """ Run random walks """

    load_data()
#    load_data("_full");
#    graph_file = sys.argv[1]
#    out_file = sys.argv[2]
#    G_data = json.load(open(graph_file))
#    G = json_graph.node_link_graph(G_data)
#    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
#    G = G.subgraph(nodes)
#    pairs = run_random_walks(G, nodes)
#    with open(out_file, "w") as fp:
#        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
