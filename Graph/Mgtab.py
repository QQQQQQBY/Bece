import matplotlib.pyplot as plt
import numpy as np

# 创建两组数据，每组数据为二维矩阵
data1 = np.array([[92.25, 91.96, 91.57, 91.86, 92.16],[92.65, 92.16, 92.16, 91.47, 91.47],[91.96, 92.16, 91.57, 91.76, 92.35],[92.06, 91.67, 92.55, 90.69, 91.67],[91.27, 91.86, 91.47, 92.84, 91.76]])
data2 = np.array([[90.31, 89.77, 89.07, 89.41, 89.75],[90.68, 90.20, 89.83, 89.11, 88.98],[89.55, 89.83, 88.93, 89.32, 90.25],[90.00, 89.43, 90.56, 88.22, 89.23],[89.01, 89.61, 89.18, 90.82, 89.35]])
# 创建画布和子图对象
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))

# 绘制第一个热力图，并设置标题
heatmap1 = ax1.imshow(data1, cmap=plt.cm.Purples, vmin=91.60, vmax=93.50)
ax1.set_title('MGTAB Acc', fontsize=30)
ax1.set_xticks(np.arange(len(data1[0])) + 0.5, ['0.01','0.1','1','2','3'], fontsize=20)
ax1.set_yticks(np.arange(len(data1)) + 0.5, ['0.01','0.1','1','2','3'], fontsize=20)
ax1.set_xlabel(r'$\lambda_{KL}$', fontsize=30)
ax1.set_ylabel(r'$\lambda_{edge}$', fontsize=30)
# 在热力图中标注数值
for i in range(len(data1)):
    for j in range(len(data1[0])):
        ax1.text(j, i, '{:.2f}'.format(data1[i][j]), ha='center', va='center', color='black', fontsize=20)
# 添加颜色条
# plt.colorbar(heatmap1, ax=ax1)
cbar = plt.colorbar(heatmap1, ax=ax1)
cbar.ax.tick_params(labelsize=20)

# 绘制第二个热力图，并设置标题
heatmap2 = ax2.imshow(data2, cmap=plt.cm.Purples, vmin=89.00, vmax=91.50)
ax2.set_title('MGTAB F1', fontsize=30)
ax2.set_xticks(np.arange(len(data1[0])) + 0.5, ['0.01','0.1','1','2','3'], fontsize=20)
ax2.set_yticks(np.arange(len(data1)) + 0.5, ['0.01','0.1','1','2','3'], fontsize=20)
ax2.set_xlabel(r'$\lambda_{KL}$', fontsize=30)
ax2.set_ylabel(r'$\lambda_{edge}$', fontsize=30)
        
for i in range(len(data2)):
    for j in range(len(data2[0])):
        ax2.text(j, i, '{:.2f}'.format(data2[i][j]), ha='center', va='center', color='black', fontsize=20)

cbar = plt.colorbar(heatmap2, ax=ax2)
cbar.ax.tick_params(labelsize=20)

# 保存图片
plt.savefig('MGTAB.png')

# 显示图像
plt.show()
