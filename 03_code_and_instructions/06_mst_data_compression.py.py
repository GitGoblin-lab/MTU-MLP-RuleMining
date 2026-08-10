import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


node_alpha = 0.5       
node_size = 1200       
node_label_size = 12   


# 1. 
file_path = '模糊相似矩阵H.xlsx'
df = pd.read_excel(file_path, index_col=0)


nodes = df.columns.tolist()

# 2. 
G = nx.Graph()


edges = []
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        weight = df.iloc[i, j]
        if pd.notna(weight):
            edges.append((nodes[i], nodes[j], weight))


G.add_weighted_edges_from(edges)

# 3. 
T = nx.maximum_spanning_tree(G)

# 4. 
plt.figure(figsize=(40, 40))


pos = nx.kamada_kawai_layout(T)


nx.draw_networkx_edges(T, pos, alpha=0.6, edge_color='gray', width=2)

nx.draw_networkx_nodes(
    T, pos,
    node_size=node_size,    
    alpha=node_alpha,       
    node_color='skyblue',
    edgecolors='white',
    linewidths=2
)


nx.draw_networkx_labels(
    T, pos,
    font_size=node_label_size,  
    font_family='sans-serif',
    font_weight='bold'
)



node_degree = dict(T.degree())

for (u, v, data) in T.edges(data=True):
    weight = round(data['weight'], 2)
    weight_label = f"{weight:.2f}"
    
    if node_degree[v] == 1:
        label_pos = 0.9
    elif node_degree[u] == 1:
        label_pos = 0.1
    else:
        label_pos = 0.5
    
    nx.draw_networkx_edge_labels(
        T, pos,
        edge_labels={(u, v): weight_label},
        label_pos=label_pos,
        font_size=8,
        font_color='darkred',
        alpha=0.9,
        font_weight='bold'
    )



plt.title('Maximum Spanning Tree of Fuzzy Similarity Matrix', fontsize=36)
plt.axis('off')


plt.savefig('maximum_spanning_tree_improved.png', bbox_inches='tight', dpi=300)
plt.show()