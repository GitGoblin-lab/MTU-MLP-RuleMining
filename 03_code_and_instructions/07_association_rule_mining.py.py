# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def extract_association_rules(input_csv, output_csv, min_supp=0.01, min_conf=0.3, max_len=6):

    print(f"{input_csv} ...")
    # 1. 
    df = pd.read_csv(input_csv)
    
    df_bool = df.astype(bool)
    
    print(f"正在挖掘频繁项集 (最小支持度={min_supp}, 最大长度={max_len}) ...")
    # 2. 

    frequent_itemsets = apriori(df_bool, min_support=min_supp, max_len=max_len, use_colnames=True)
    
    if frequent_itemsets.empty:
        print("未找到满足当前支持度阈值的频繁项集，请尝试调低 min_supp。")
        return
        
    print(f"正在生成关联规则 (最小置信度={min_conf}) ...")
    # 3. 
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
    
    if rules.empty:
        print("未找到满足当前置信度阈值的关联规则，请尝试调低 min_conf。")
        return

    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    

    rules = rules.sort_values(by=['confidence', 'lift'], ascending=[False, False])
    
    # 4. 
    rules.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"成功提取 {len(rules)} 条关联规则！")
    print(f"规则已保存至当前目录下的: {output_csv}")

if __name__ == "__main__":
    INPUT_FILE = 'H.csv'          
    OUTPUT_FILE = 'association_rules.csv' 
    
    SUPP = 0.01  
    CONF = 0.3  
    MAXLEN = 6  


    extract_association_rules(
        input_csv=INPUT_FILE, 
        output_csv=OUTPUT_FILE, 
        min_supp=SUPP, 
        min_conf=CONF, 
        max_len=MAXLEN
    )