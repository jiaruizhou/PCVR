# encoding=utf-8
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"]=['SimHei']
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams.update({'font.size': 12})

plt.figure(figsize=(15,8))
grid = plt.GridSpec(9,17, wspace=0.1, hspace=0.1)

colors = [(238,238,238)]
normalized_colors = [(r/255, g/255, b/255) for r, g, b in colors]

plt.subplot(grid[0:4,0:7], facecolor=normalized_colors[0])
# plt.subplot(2,2,2)
x_1 = [1,2,3,4,5,6,8,10,16,20]
B_1 = [80.45, 80.56, 80.71, 80.69, 80.62, 80.45, 80.94, 80.33, 79.34, 79.28]
N_1 = [74.69, 74.83, 74.65, 74.37, 74.68, 73.35, 74.89, 74.17, 73.37, 72.22]
H_1 = [77.47, 77.59, 77.56, 77.40, 77.53, 76.74, 77.80, 77.13, 76.23, 75.58]
x_1_indexes = range(len(x_1))

plt.plot(x_1_indexes, B_1, linewidth =1.5, label = 'Base', marker='s',linestyle='-')
plt.plot(x_1_indexes, H_1, linewidth =1.5, label = 'H', marker='*',linestyle=':')
plt.plot(x_1_indexes, N_1, linewidth =1.5, label = 'New', marker='o',linestyle='-.')
plt.xticks(x_1_indexes, x_1)

plt.legend(fontsize=10)
plt.grid(True, color='white')  # 设置网格线为白色
plt.xlabel('(a) Vision prompt length', fontsize=14)
plt.ylabel('Score(%)', fontsize=14)



plt.subplot(grid[0:4,8:15], facecolor=normalized_colors[0])
# plt.subplot(2,15,16)
x_2 = [1,2,3,4,5,6,8,10,12,16,20]
B_2 = [80.61, 80.69, 79.72, 78.79, 80.94, 80.34, 80.54, 79.54, 80.32, 80.47, 80.08]
N_2 = [74.08, 74.68, 73.12, 72.51, 74.89, 72.98, 74.08, 72.33, 73.66, 73.71, 73.12]
H_2 = [77.21, 77.57, 76.28, 75.52, 77.80, 76.49, 77.18, 75.76, 76.85, 76.94, 76.44]
x_2_indexes = range(len(x_2))

plt.plot(x_2_indexes, B_2, linewidth =1.5, label = 'Base', marker='s',linestyle='-')
plt.plot(x_2_indexes, H_2, linewidth =1.5, label = 'H', marker='*',linestyle=':')
plt.plot(x_2_indexes, N_2, linewidth =1.5, label = 'New', marker='o',linestyle='-.')

plt.xticks(x_2_indexes, x_2)

plt.legend(fontsize=8)
plt.grid(True, color='white')  # 设置网格线为白色
plt.xlabel('(b) Text prompt length', fontsize=14)
plt.ylabel('Score(%)', fontsize=14)



plt.subplot(grid[5:9,0:7], facecolor=normalized_colors[0])
# plt.subplot(2,2,1)
x_3 = ['1', '1-3', '1-5', '1-7', '1-9', '1-12']
B_3 = [78.18, 79.67, 79.69, 80.52, 80.94, 78.17]
N_3 = [71.90, 72.30, 73.72, 74.72, 74.89, 71.60]
H_3 = [74.91, 75.80, 76.59, 77.51, 77.80, 74.74]
x_3_indexes = range(len(x_3))

plt.plot(x_3_indexes, B_3, linewidth =1.5, label = 'Base', marker='s',linestyle='-')
plt.plot(x_3_indexes, H_3, linewidth =1.5, label = 'H', marker='*',linestyle=':')
plt.plot(x_3_indexes, N_3, linewidth =1.5, label = 'New', marker='o',linestyle='-.')
plt.xticks(x_3_indexes, x_3)

plt.legend(fontsize=9)
plt.grid(True, color='white')  # 设置网格线为白色
plt.xlabel('(c) Vision prompt layer', fontsize=14)
plt.ylabel('Score(%)', fontsize=14)



plt.subplot(grid[5:9,8:15], facecolor=normalized_colors[0])
# plt.subplot(2,2,1)
x_4 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
B_4 = [78.82, 79.61, 79.45, 79.46, 79.79, 80.31, 80.57, 80.41, 80.90, 80.94]
N_4 = [72.32, 73.53, 74.28, 74.11, 73.79, 74.95, 74.65, 74.56, 74.64, 74.89]
H_4 = [75.43, 76.45, 76.78, 76.69, 76.67, 77.54, 77.49, 77.37, 77.65, 77.80]
x_4_indexes = range(len(x_4))

plt.plot(x_4_indexes, B_4, linewidth =1.5, label = 'Base', marker='s',linestyle='-')
plt.plot(x_4_indexes, H_4, linewidth =1.5, label = 'H', marker='*',linestyle=':')
plt.plot(x_4_indexes, N_4, linewidth =1.5, label = 'New', marker='o',linestyle='-.')
plt.xticks(x_4_indexes, x_4)

plt.legend(fontsize=10)
plt.grid(True, color='white')  # 设置网格线为白色
plt.xlabel('(d) Loss weighting factor', fontsize=14)
plt.ylabel('Score(%)', fontsize=14)

plt.tight_layout()
plt.savefig('abla.pdf', bbox_inches='tight')
plt.show()