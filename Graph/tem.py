
# from matplotlib import pyplot as plt
# from matplotlib import colorbar
# import seaborn as sns
# import numpy as np
# import pandas as pd

# acc = np.array([[87.79, 87.53, 87.67, 87.60, 87.72],[87.33, 87.43, 87.72, 87.82, 87.67],[87.57, 87.57, 87.67, 87.72, 87.77],[87.57, 87.26, 87.53, 87.57, 87.53],[87.64, 87.87, 87.53, 87.67, 87.67]])
# sns.set(font_scale=1.5)

# data = pd.DataFrame(
#     data=acc, 
#     columns=['0.01','0.1','1','2','3'], 
#     index=['0.01','0.1','1','2','3']
# )
# cmap = sns.heatmap(data,linewidths=0.8,annot=True,fmt="d")
# cmap = sns.heatmap(data,linewidths=0.8,annot=True,fmt="d")
# plt.xlabel("lambda_KL",size=20)
# plt.ylabel("lambda_e",size=20,rotation=0)
# plt.title("Twibot-20 Acc",size=20)

# cbar = cmap.collections[0].colorbar
# cbar.ax.tick_params(labelsize=20,labelcolor="blue")
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # 创建一个3x3的矩阵作为数据
# data = np.array([[87.79, 87.53, 87.67, 87.60, 87.72],[87.33, 87.43, 87.72, 87.82, 87.67],[87.57, 87.57, 87.67, 87.72, 87.77],[87.57, 87.26, 87.53, 87.57, 87.53],[87.64, 87.87, 87.53, 87.67, 87.67]])

# # 绘制热力图，并设置颜色映射范围为0到1
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# # plt.subplot(1, 2, 1)  # 分别指定：1行，2列，第1列
# heatmap = plt.pcolor(data, cmap=plt.cm.Blues, vmin=87.20, vmax=88.00)

# # 显示热力图的值
# for i in range(len(data)):
#     for j in range(len(data[0])):
#         plt.text(j + 0.5, i + 0.5, f'{data[i][j]:.2f}', ha='center', va='center')

# # 添加坐标轴标签和标题
# plt.xticks(np.arange(len(data[0])) + 0.5, ['0.01','0.1','1','2','3'])
# plt.yticks(np.arange(len(data)) + 0.5, ['0.01','0.1','1','2','3'])
# plt.xlabel(r'$\lambda_{KL}$')
# plt.ylabel(r'$\lambda_{edge}$')
# plt.title('Twibot-20 Acc')

# # 添加颜色条
# plt.colorbar(heatmap)


# data1 = np.array([[90.14, 90.02, 90.00, 90.04,90.06],[89.85, 89.70, 90.08, 90.20, 90.04],[90.04, 89.95, 90.05, 90.09, 90.13],[90.09, 89.86, 90.01, 90.03, 89.91],[90.06, 90.30, 90.05, 90.13, 90.06]])
# # plt.subplot(1, 2, 2)  # 分别指定：1行，2列，第2列

# heatmap1 = plt.pcolor(data1, cmap=plt.cm.Blues, vmin=89.80, vmax=90.20)

# # 显示热力图的值
# for i in range(len(data1)):
#     for j in range(len(data1[0])):
#         plt.text(j + 0.5, i + 0.5, f'{data1[i][j]:.2f}', ha='center', va='center')

# # 添加坐标轴标签和标题
# plt.xticks(np.arange(len(data1[0])) + 0.5, ['0.01','0.1','1','2','3'])
# plt.yticks(np.arange(len(data1)) + 0.5, ['0.01','0.1','1','2','3'])
# plt.xlabel(r'$\lambda_{KL}$')
# plt.ylabel(r'$\lambda_{edge}$')
# plt.title('Twibot-20 F1')

# # 添加颜色条
# plt.colorbar(heatmap1)

# # 保存图片
# plt.savefig('heatmap.png', dpi=300)

# # 显示绘制的热力图
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# 创建两组数据，每组数据为二维矩阵
data1 = np.array([[87.79, 87.53, 87.67, 87.60, 87.72],[87.33, 87.43, 87.72, 87.82, 87.67],[87.57, 87.57, 87.67, 87.72, 87.77],[87.57, 87.26, 87.53, 87.57, 87.53],[87.64, 87.87, 87.53, 87.67, 87.67]])
data2 = np.array([[90.14, 90.02, 90.00, 90.04,90.06],[89.85, 89.70, 90.08, 90.20, 90.04],[90.04, 89.95, 90.05, 90.09, 90.13],[90.09, 89.86, 90.01, 90.03, 89.91],[90.06, 90.30, 90.05, 90.13, 90.06]])
# 创建画布和子图对象
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))

# 绘制第一个热力图，并设置标题
heatmap1 = ax1.imshow(data1, cmap=plt.cm.Blues, vmin=87.20, vmax=88.00)
ax1.set_title('Twibot-20 Acc', fontsize=30)
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
heatmap2 = ax2.imshow(data2, cmap=plt.cm.Blues, vmin=89.80, vmax=90.40)
ax2.set_title('Twibot-20 F1', fontsize=30)
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
plt.savefig('heatmap.png')

# 显示图像
plt.show()
