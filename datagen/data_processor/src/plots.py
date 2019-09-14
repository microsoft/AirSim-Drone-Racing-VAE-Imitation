import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as plt

# note on confidence interval:
# ci=95 shows 95% CI
# ci='sd' shows STD

# load data
data_dir = '/home/rb/data/temp/data_full_new.csv'
table = pd.read_csv(data_dir)
# g = sns.relplot(x="Noise level [m]", y="Num gates passed", kind="line", hue='Method name', data=table)
# bla = sns.lineplot(x="Noise level [m]", y="Num gates passed", hue='Method name', data=table, ci=10)
# bla = sns.boxplot(x="Noise level [m]", y="Num gates passed", hue='Method name', data=table)
order_list = ['bc_con','bc_unc','bc_reg','bc_full','bc_img']
sns.set(style="whitegrid")
fig = sns.barplot(x="Noise level [m]", y="Performance", hue='Method name', data=table, hue_order=order_list, capsize=.05, ci=90)
fig.set_title('Performance of drone racing policies')
fig.set_ylabel('Performance [%]')
fig.set_xlabel('Gate displacement amplitude [m]')
plt.pyplot.show()
