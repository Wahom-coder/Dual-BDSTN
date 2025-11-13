
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data_for_TPM.csv')
Machine_name = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13']
TS = []
TB = []
row_index = -3
for i in range(len(Machine_name)):
    TS.append(float(df.iloc[row_index, 2 * i])*100/float((24 * 60)))
    TB.append(float(df.iloc[row_index, 2 * i + 1])*100/float((24 * 60)))
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['figure.dpi'] = 1200
plt.rcParams['figure.figsize'] = (10, 3)
#
# #将横坐标国家转换为数值
# x = np.arange(len(Machine_name))
# width = 0.2
#
# TS_x = x
# TB_x = x + width
#
# plt.bar(TS_x,TS,width=width,color="gold",label="TS")
# plt.bar(TB_x,TB,width=width,color="silver",label="TB")
#
# plt.xticks(x + width, labels=Machine_name)
#
# # #显示柱状图的高度文本
# # for i in range(len(Mahcine_name)):
# #     plt.text(TS_x[i],TS[i], TS[i],va="bottom",ha="center",fontsize=8)
# #     plt.text(TB[i],TB[i], TS[i],va="bottom",ha="center",fontsize=8)
#
# plt.show()
# 标签
plt.rcParams.update({
    'legend.fontsize': 10  # 全局设置图例字体大小
})

x = np.arange(len(Machine_name))  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots()
rects2 = ax.bar(x - width/2, TS, width, label='% TS', color='#6270B7' )
rects1 = ax.bar(x + width/2, TB, width, label='% TB', color='#B83945')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percentage of time', fontsize=12, weight='bold')
ax.set_xlabel('Machine', fontsize=12, weight='bold')
# string = str(row_index)
# ax.set_title(string)
ax.set_title('Blockage and Starvation times in a manufacturing line', fontsize=12, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(Machine_name, fontsize=12)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)
bwith = 1.0 # 边框宽度设置为2
ax = plt.gca()  # 获取边框
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.tick_params(width=1.0)#修改刻度线线粗细width参数，修改刻度字体labelsize参数
fig.tight_layout()
plt.savefig('F:/第一篇论文_预测/revised data/bn.png', format='png', dpi=1200)
# plt.show()


def Bn_identification_for_the_fisrt(TS_1, TB_1, TS_2, TB_2):
    if float(TB_1 - TS_1) > 0 \
            and float(TB_2 - TS_2) < 0 \
            and float(TB_1 + TS_1) < float(TS_2 + TB_2):
    # 条件满足时的代码
        return True

def Bn_identification_for_the_Last(TS_n_1, TB_n_1, TS_n, TB_n):
    if float(TB_n_1 - TS_n_1) > 0 and float(TB_n - TS_n) < 0 and float(TB_n + TS_n) < float(TS_n_1 +TB_n_1):
        return True

def Bn_identification_for_the_Other(TS_j_1, TB_j_1, TS_j, TB_j, TS_j_2, TB_j_2):
    if float(TB_j_1 - TS_j_1) > 0 and float(TB_j_2 - TS_j_2) < 0 and float(TB_j + TS_j) < float(TS_j_1 + TB_j_1) and float(TB_j + TS_j) < float(TS_j_2 + TB_j_2):
        return True


Bn = []

value_1 = "M1"
value_2 = "M2"
value_1_index = Machine_name.index(value_1)
value_2_index = Machine_name.index(value_2)
if Bn_identification_for_the_fisrt(TS[value_1_index], TB[value_1_index],TS[value_2_index], TB[value_2_index])==True:
    Bn.append(value_1)

value_1 = "M3"
value_2 = "M2"
value_1_index = Machine_name.index(value_1)
value_2_index = Machine_name.index(value_2)
if Bn_identification_for_the_fisrt(TS[value_1_index], TB[value_1_index],TS[value_2_index], TB[value_2_index])==True:
    Bn.append(value_1)

value_1 = "M1"
value_2 = "M2"
value_3 = "M4"
value_1_index = Machine_name.index(value_1)
value_2_index = Machine_name.index(value_2)
value_3_index = Machine_name.index(value_3)
if Bn_identification_for_the_Other(TS[value_1_index], TB[value_1_index],
                                   TS[value_2_index], TB[value_2_index],
                                   TS[value_3_index], TB[value_3_index])==True:
    Bn.append(value_2)

value_1 = "M3"
value_2 = "M2"
value_3 = "M4"
value_1_index = Machine_name.index(value_1)
value_2_index = Machine_name.index(value_2)
value_3_index = Machine_name.index(value_3)
if Bn_identification_for_the_Other(TS[value_1_index], TB[value_1_index],
                                   TS[value_2_index], TB[value_2_index],
                                   TS[value_3_index], TB[value_3_index])==True:
    Bn.append(value_2)

value_1 = "M5"
value_2 = "M4"
value_1_index = Machine_name.index(value_1)
value_2_index = Machine_name.index(value_2)
if Bn_identification_for_the_fisrt(TS[value_1_index], TB[value_1_index],TS[value_2_index], TB[value_2_index])==True:
    Bn.append(value_1)

value_1 = "M2"
value_2 = "M4"
value_3 = "M6"
value_1_index = Machine_name.index(value_1)
value_2_index = Machine_name.index(value_2)
value_3_index = Machine_name.index(value_3)
if Bn_identification_for_the_Other(TS[value_1_index], TB[value_1_index],
                                   TS[value_2_index], TB[value_2_index],
                                   TS[value_3_index], TB[value_3_index])==True:
    Bn.append(value_2)

value_1 = "M5"
value_2 = "M4"
value_3 = "M6"
# value_1_index = Machine_name.index(value_1)
# value_2_index = Machine_name.index(value_2)
# value_3_index = Machine_name.index(value_3)
if Bn_identification_for_the_Other(TS[value_1_index], TB[value_1_index],
                                   TS[value_2_index], TB[value_2_index],
                                   TS[value_3_index], TB[value_3_index])==True:
    Bn.append(value_2)

value_1 = "M4"
value_2 = "M6"
value_3 = "M7"
value_1_index = Machine_name.index(value_1)
value_2_index = Machine_name.index(value_2)
value_3_index = Machine_name.index(value_3)
if Bn_identification_for_the_Other(TS[value_1_index], TB[value_1_index],
                                   TS[value_2_index], TB[value_2_index],
                                   TS[value_3_index], TB[value_3_index])==True:
    Bn.append(value_2)

value_1 = "M6"
value_2 = "M7"
value_3 = "M8"
value_1_index = Machine_name.index(value_1)
value_2_index = Machine_name.index(value_2)
value_3_index = Machine_name.index(value_3)
a = TB[value_1_index]-TS[value_1_index]
b = TB[value_3_index]-TS[value_3_index]
c = TB[value_1_index]+TS[value_1_index]
d = TB[value_2_index]+TS[value_2_index]
e = TB[value_3_index]+TS[value_3_index]
if Bn_identification_for_the_Other(TS[value_1_index], TB[value_1_index],
                                   TS[value_2_index], TB[value_2_index],
                                   TS[value_3_index], TB[value_3_index])==True:
    Bn.append(value_2)

value_1 = "M7"
value_2 = "M8"
value_3 = "M9"
value_1_index = Machine_name.index(value_1)
value_2_index = Machine_name.index(value_2)
value_3_index = Machine_name.index(value_3)
if Bn_identification_for_the_Other(TS[value_1_index], TB[value_1_index],
                                   TS[value_2_index], TB[value_2_index],
                                   TS[value_3_index], TB[value_3_index])==True:
    Bn.append(value_2)

value_1 = "M8"
value_2 = "M9"
value_3 = "M10"
value_1_index = Machine_name.index(value_1)
value_2_index = Machine_name.index(value_2)
value_3_index = Machine_name.index(value_3)
if Bn_identification_for_the_Other(TS[value_1_index], TB[value_1_index],
                                   TS[value_2_index], TB[value_2_index],
                                   TS[value_3_index], TB[value_3_index])==True:
    Bn.append(value_2)

value_1 = "M9"
value_2 = "M10"
value_3 = "M11"
value_1_index = Machine_name.index(value_1)
value_2_index = Machine_name.index(value_2)
value_3_index = Machine_name.index(value_3)
if Bn_identification_for_the_Other(TS[value_1_index], TB[value_1_index],
                                   TS[value_2_index], TB[value_2_index],
                                   TS[value_3_index], TB[value_3_index])==True:
    Bn.append(value_2)

value_1 = "M10"
value_2 = "M11"
value_3 = "M12"
value_1_index = Machine_name.index(value_1)
value_2_index = Machine_name.index(value_2)
value_3_index = Machine_name.index(value_3)
if Bn_identification_for_the_Other(TS[value_1_index], TB[value_1_index],
                                   TS[value_2_index], TB[value_2_index],
                                   TS[value_3_index], TB[value_3_index])==True:
    Bn.append(value_2)

value_1 = "M11"
value_2 = "M12"
value_1_index = Machine_name.index(value_1)
value_2_index = Machine_name.index(value_2)
if Bn_identification_for_the_Last(TS[value_1_index], TB[value_1_index],TS[value_2_index], TB[value_2_index])==True:
    Bn.append(value_2)

value_1 = "M13"
value_2 = "M12"
value_1_index = Machine_name.index(value_1)
value_2_index = Machine_name.index(value_2)
if Bn_identification_for_the_Last(TS[value_1_index], TB[value_1_index],TS[value_2_index], TB[value_2_index])==True:
    Bn.append(value_2)

value_1 = "M13"
value_2 = "M12"
value_1_index = Machine_name.index(value_1)
value_2_index = Machine_name.index(value_2)
if Bn_identification_for_the_fisrt(TS[value_1_index], TB[value_1_index],TS[value_2_index], TB[value_2_index])==True:
    Bn.append(value_1)
print('over')





