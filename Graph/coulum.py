import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
acc = [95.95,95.48,95.76,96.05,95.67,96.70,96.33,96.89,96.42,96.89]
f1 = [96.69,96.28,96.56,96.78,96.50,97.30,96.98,97.46,97.09,97.46]
x = ['10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
x_len = np.arange(len(x))
plt.figure(figsize=(25, 8))
ax = plt.axes()
plt.grid(axis="y", c='#d2c9eb', linestyle = '--',zorder=0)


plt.bar(x=x, height=acc, label='Acc', color='#5d84ef', alpha=0.8,linewidth = 2,edgecolor='black')
plt.xticks(x_len, x, fontproperties='Times New Roman',fontsize = 30)
plt.yticks(fontproperties='Times New Roman',fontsize = 30)
plt.ylim(ymin=92)
plt.xlabel("Data Percentage", fontproperties='Times New Roman',fontsize=45)
# plt.ylabel("Twibot-20",fontproperties='Times New Roman', fontsize=45)
plt.ylabel("MGTAB",fontproperties='Times New Roman', fontsize=45)
plt.plot(x, f1, "r", marker='o', c='#93211e', ms=10, linewidth='3', label="f1")
# 显示数字
for a, b in zip(x, f1):
    plt.text(a, b + 0.3, b, ha='center', va='bottom', fontsize=35)

plt.legend(ncol=2,prop={'family' : 'Times New Roman', 'size': 35})
# plt.legend(prop={'family' : 'Times New Roman', 'size': 35}, ncol = 2)

ax.spines['bottom'].set_linewidth('2.0')#设置边框线宽为2.0
ax.spines['bottom'].set_color('5d84ef')
ax.spines['top'].set_linewidth('2.0')#设置边框线宽为2.0
ax.spines['top'].set_color('black')
ax.spines['right'].set_linewidth('2.0')#设置边框线宽为2.0
ax.spines['right'].set_color('black')
ax.spines['left'].set_linewidth('2.0')#设置边框线宽为2.0
ax.spines['left'].set_color('black')
plt.tight_layout()
plt.savefig('column_scale_MGTAB.png')
plt.show()
 