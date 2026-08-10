# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np
import skfuzzy as fuzz

pd.set_option('display.max_columns', None)    
pd.set_option('display.width', 1000)         
pd.set_option('display.float_format', lambda x: f'{x:.4f}')  


# ===================== 1===================
df = pd.read_excel("聚类数据.xlsx")  

sample_id = df.iloc[:, 0]          
feature_names = df.columns[1:]     
data_matrix = df.iloc[:, 1:].values  

data_T = data_matrix.T


# ===================== 2.=====================
n_clusters = 3       
m = 2               
error = 0.001        
max_iter = 2000     


# ===================== 3.=====================
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data=data_T,
    c=n_clusters,
    m=m,
    error=error,
    maxiter=max_iter,
    init=None
)


# ===================== 4. =====================
cluster_centers = pd.DataFrame(
    cntr,
    columns=feature_names,
    index=[f"聚类中心_{i+1}" for i in range(n_clusters)]
)


hard_labels = np.argmax(u, axis=0) + 1  
sample_result = pd.DataFrame({"样本ID": sample_id, "聚类类别": hard_labels})
for i in range(n_clusters):
    sample_result[f"隶属度_类别{i+1}"] = u[i]

# ===================== 5. =====================
print("=" * 60)
print("模糊C-均值聚类完成")
print(f"聚类数目：{n_clusters}")
print(f"模糊划分系数(FPC)：{fpc:.4f}  （越接近1，聚类效果越好）")
print("=" * 60)

print("\n【最终聚类中心矩阵】")
print(cluster_centers)  

print("\n【样本聚类结果（前10行）】")
print(sample_result.head(10))


