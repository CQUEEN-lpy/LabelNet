"""
import
"""
import pandas as pd
import numpy as np
import random

"""
extract certain information from the logging csv and generate a sz string to extract images using the linux terminal
"""

task_name = 'man_top_color'
csv_path = '/home/zhang.xinxi/CV/data/alibaba/csv/predict/{name}/{name}.csv'.format(name=task_name)
df = pd.read_csv(csv_path)
df = np.array(df)
table = {}
count = 0
s = 'sz'
for i in df:
    if count == 100:
        pass
    rd = random.random()
    if 'micai' in i[3] and 'wh' not in i[3] and i[2]>0.7:
        s += ' ' + i[0]
        count += 1
print(s)
print(count)
