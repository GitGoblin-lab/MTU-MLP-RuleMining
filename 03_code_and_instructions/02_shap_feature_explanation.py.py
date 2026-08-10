# -*- coding: utf-8 -*-
"""

"""

# 1. 
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  
import shap
import numpy as np  


plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False  

# 2. 

data = pd.read_csv('MTU_H.csv')

# 3. 
cols = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 
        'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6']

# 4. 
model = xgb.XGBRegressor(
    max_depth=5, 
    learning_rate=0.191, 
    n_estimators=76, 
    min_child_weight=3.546,
    subsample=0.822,
    colsample_bytree=0.500,
    reg_alpha=0.081,
    reg_lambda=0.645,
    base_score=0.5
)
model.fit(data[cols], data['y'].values)

# 5. 
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data[cols])

print("SHAP values shape:", shap_values.shape)
y_base = explainer.expected_value
print("Explainer expected value (base line):", y_base)

# 6. 
data['pred'] = model.predict(data[cols])
print("Mean of predictions:", data['pred'].mean())

# 7. 
shap.summary_plot(shap_values, data[cols], max_display=26)
plt.show()

# 8. 
shap.summary_plot(shap_values, data[cols], plot_type="bar", max_display=26)
plt.show()

# 9. 
feat_x = 'N6'
feat_inter = 'R12'

fig, ax = plt.subplots(figsize=(3.5, 2.2), dpi=300)

shap.dependence_plot(
    feat_x, 
    shap_values, 
    data[cols], 
    interaction_index=feat_inter, 
    dot_size=8,        
    alpha=0.9,         
    ax=ax, 
    show=False
)


hist_alpha = 0.35  
bins_count = 25    

counts, bin_edges = np.histogram(data[feat_x].dropna(), bins=bins_count)
ymin, ymax = ax.get_ylim()
y_range = ymax - ymin

scaled_counts = (counts / counts.max()) * (y_range * 0.2)

ax.bar(bin_edges[:-1], scaled_counts, width=np.diff(bin_edges), 
       align='edge', color='darkgray', alpha=hist_alpha, zorder=0, bottom=ymin)

ax.set_ylim(ymin, ymax)

ax.axhline(0, color='gray', linestyle='--', linewidth=0.35, zorder=0.7)


all_axes = fig.get_axes()
if len(all_axes) > 1:
    cb_ax = all_axes[-1]  
    cb_ax.tick_params(labelsize=6) 
    cb_ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=9))
    cb_ax.set_ylabel(feat_inter, fontsize=7)


# ax.set_title(f'Dependence plot between {feat_x} and {feat_inter}', fontsize=8, pad=8)
ax.set_ylabel(f'SHAP value for\n{feat_x}', fontsize=7.5)
ax.set_xlabel(feat_x, fontsize=7.5)
ax.tick_params(labelsize=5.5)
ax.grid(True, linestyle='-', alpha=0.4, linewidth=0.8)

plt.tight_layout()
plt.show()