import pandas as pd
import numpy as np

"""
if we use all 4 GPU to predict, we get 4 csv, this py combine them to a uniform csv
"""

task_name = 'man_top_style'
csv_path = '/home/zhang.xinxi/CV/data/alibaba/csv/predict/{name}/{name}.csv_'.format(name=task_name)
total = None
for i in range(1,5):
    path = csv_path + str(i) + '.csv'
    df = np.array(pd.read_csv(path))
    if total is None:
        total = df
    else:
        total = np.concatenate((total,df),axis=0)

total = pd.DataFrame(total).set_index(0)
total.to_csv('/home/zhang.xinxi/CV/data/alibaba/csv/predict/{name}/{name}'.format(name=task_name) + '.csv')
