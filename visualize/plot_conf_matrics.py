from sklearn.metrics import confusion_matrix
from re import match
# from models.model import PARAMS
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import numpy as np


# classes = PARAMS['data']['classes'][1]


def conf_matrix(trues, preds, dropzero=True, perc=False, rounded=True,
                xrot=None):
    x = confusion_matrix(trues, preds, classes)
    if perc:
        x = np.round(x / len(preds) * 100)
        if rounded:
            x = np.round(x)
    df_cm = pd.DataFrame(x, classes, classes)
    pd.set_option('precision', 0)
    if (dropzero):
        df_cm = df_cm.drop([c for c in classes if c not in trues])
    # plt.figure(figsize=(10,7))
    plt.clf()
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},
               cbar_kws={'format': '%.0f'})  # font size
    plt.yticks(rotation=0)
    if xrot is not None:
        plt.xticks(rotation=xrot)
    plt.tight_layout()
    return df_cm


def fasta_cm(fastafile):
    y_trues = []
    y_preds = []
    with open(fastafile) as f:
        for line in f.readlines():
            if line.startswith('>'):
                y_true, y_pred = match(r'>([^;]+); predicted: (.*)$', line
                                       ).groups()
                y_trues.append(y_true)
                y_preds.append(y_pred)
    conf_matrix(y_trues, y_preds)
    plt.savefig(fastafile + '_conf_matrix.png')

def plot_bar():
    
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(13, 4))
    # 构造x轴刻度标签、数据
    labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    first = [20, 34, 30, 35, 27]
    second = [25, 32, 34, 20, 25]
    third = [21, 31, 37, 21, 28]
    fourth = [26, 31, 35, 27, 21]

    # 两组数据
    plt.subplot(131)
    x = np.arange(len(labels))  # x轴刻度标签位置
    width = 0.25  # 柱子的宽度
    # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
    # x - width/2，x + width/2即每组数据在x轴上的位置
    plt.bar(x - width/2, first, width, label='1')
    plt.bar(x + width/2, second, width, label='2')
    plt.ylabel('Scores')
    plt.title('2 datasets')
    # x轴刻度标签位置不进行计算
    plt.xticks(x, labels=labels)
    plt.legend()
    # 三组数据
    plt.subplot(132)
    x = np.arange(len(labels))  # x轴刻度标签位置
    width = 0.25  # 柱子的宽度
    # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
    # x - width，x， x + width即每组数据在x轴上的位置
    plt.bar(x - width, first, width, label='1')
    plt.bar(x, second, width, label='2')
    plt.bar(x + width, third, width, label='3')
    plt.ylabel('Scores')
    plt.title('3 datasets')
    # x轴刻度标签位置不进行计算
    plt.xticks(x, labels=labels)
    plt.legend()
    # 四组数据
    plt.subplot(133)
    x = np.arange(len(labels))  # x轴刻度标签位置
    width = 0.2  # 柱子的宽度
    # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
    plt.bar(x - 1.5*width, first, width, label='1')
    plt.bar(x - 0.5*width, second, width, label='2')
    plt.bar(x + 0.5*width, third, width, label='3')
    plt.bar(x + 1.5*width, fourth, width, label='4')
    plt.ylabel('Scores')
    plt.title('4 datasets')
    # x轴刻度标签位置不进行计算
    plt.xticks(x, labels=labels)
    plt.legend()

    plt.savefig()

def plot_stack(dataset="similar"):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    # set font of all plots to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    # set `large` font size of plots to 12
    plt.rcParams['font.size'] = 14
    plt.rcParams['figure.figsize'] = [6.4, 3.6]


    # Read the Excel file
    # xls = pd.ExcelFile('/data5/zhoujr/mae/visualize/test_result.xlsx')

    # # Get the names of all worksheets
    # sheet_names = xls.sheet_names
    # print(sheet_names)

    # 读取 Excel 文件的特定 sheet
    sheet_name = dataset  # 或者使用 sheet_name=0 表示按索引读取第一个 sheet
    if dataset=="final":
        df = pd.read_csv('stack_data_{}.csv'.format(dataset),dtype={'superkingdom': float,'phylum': float,'genus': float})
    else:
        df = pd.read_csv('stack_data_{}.csv'.format(dataset),dtype={'superkingdom': float,'phylum': float})

    # data=df.values.tolist()
    setting = df.iloc[:, 0]

    
    N = len(setting)
    if dataset =="final":
        # data_df = df.iloc[:, 1:4]
        S = df.iloc[:, 1]
        C = df.iloc[:, 2]
        M = df.iloc[:, 3]
    else:
        S = df.iloc[:, 1]
        C = df.iloc[:, 2]
        # data_df = df.iloc[:, 1:3]
        
    
    d=[]
    

    for i in range(0,len(S)):
        sum = S[i] + C[i]
        d.append(sum)
    
    #menStd = (2, 3, 4, 1, 2)
    #womenStd = (3, 5, 2, 3, 3)
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence
    
    p1 = plt.bar(ind, S, width, color='#d62728')#, yerr=menStd)
    p2 = plt.bar(ind, C, width, bottom=S)#, yerr=womenStd)
    
    if dataset=="final":
        p3 = plt.bar(ind, M, width, bottom=d)
    
    plt.ylabel('Scores')
    plt.title('Scores by group and gender')
    plt.xticks(ind, setting)
    plt.yticks(np.arange(0, 100, 20))
    if dataset=="final":
        plt.legend((p1[0], p2[0], p3[0]), ('superkingdom', 'phylum'))
    else:
        plt.legend((p1[0], p2[0]), ('superkingdom', 'phylum', 'genus'))
    
    plt.savefig('result_stack_{}.pdf'.format(dataset), bbox_inches='tight')

if __name__ == '__main__':
    # fasta_cm('bert_nc_toy_finetuned_test_fragments_1k.txt_seqs.fasta')
    # fasta_cm('bert_nc_toy_finetuned_test_coding_seqs_10k_1500nt.txt_seqs.fasta')
    plot_stack("similar")
    plot_stack("non_similar")
    plot_stack("final")