import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math

data = pd.read_csv("/Users/kpentchev/data/ner_2019_04_05_fixed.csv", encoding="utf-8", delimiter='\t', quoting=3)

'''
i = 1
for row in data.values:
    i += 1
    if not isinstance(row[3], str) and math.isnan(row[3]):
        print(i)
        print(row)

'''
tags = {}

for val in tqdm(data["Tag"].values):
    if val != 'O' and val != math.nan:
        n = tags.get(val, 0)
        tags[val] = n + 1

print(tags)

plt.rcdefaults()

y_pos = np.arange(len(tags.keys()))

bars = plt.bar(y_pos, tags.values(), align='center', alpha=0.5)

plt.xticks(y_pos, tags.keys())
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, 20 + height, '{}'.format(height), ha='center', va='bottom')


plt.ylabel('N occurence')
plt.xlabel('Tags')
plt.title('Tag distributions')

plt.show()
