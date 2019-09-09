import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as plt

# note on confidence interval:
# ci=95 shows 95% CI
# ci='sd' shows STD

# load data
data_dir = '/home/rb/data/temp/data_full.csv'
table = pd.read_csv(data_dir)
# g = sns.relplot(x="Noise level [m]", y="Num gates passed", kind="line", hue='Method name', data=table)
bla = sns.lineplot(x="Noise level [m]", y="Num gates passed", hue='Method name', data=table, ci=None)

plt.pyplot.show()
