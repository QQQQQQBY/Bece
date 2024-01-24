import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
acc = [85.70, 87.73]
f1 = [87.69,90.20]
x = ['wo_GNN','BRUE(RGT)']
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
 