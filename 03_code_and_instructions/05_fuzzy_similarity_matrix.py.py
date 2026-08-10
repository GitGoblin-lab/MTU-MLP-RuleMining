import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform


df = pd.read_csv('模糊化数据表--H.csv', index_col=0)


m = df.shape[1] # m = 8
distances = pdist(df.values, metric='euclidean') 
adjusted_distances = distances / np.sqrt(m)
dist_matrix = squareform(adjusted_distances)


np.fill_diagonal(dist_matrix, 1.0)


R_df = pd.DataFrame(dist_matrix, index=df.index, columns=df.index)
output_filename = '模糊相似矩阵R_匹配您的公式计算结果.csv'
R_df.to_csv(output_filename)

print("文件已成功导出为：{output_filename}")