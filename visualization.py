import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
# import opts
warnings.filterwarnings('ignore')

def visual(visual_cline):
    # 提取文件内容为数组 第一行作为行索引，第一列作为列索引
    #visual_matrix = pd.read_csv(visual_cline, index_col=0, header=0)
    # visual_matrix = np.array(df)

    # 生成热力图
    fig, ax = plt.subplots(figsize=(30, 4))

    # 使用seaborn包绘制热图 center:设置红蓝分界线    annot：是否显示数值注释   fmt：format的缩写，设置数值的格式化形式
    sns.heatmap(visual_cline, cmap='RdBu_r', center=0.5, annot=False, cbar=True, ax=ax)

    # 设置热图的标题和坐标轴标签
    plt.title('Heatmap of gene ')
    plt.xlabel('HCT-116')
    plt.ylabel('Drug')

    # 显示热图
    plt.show()
    return 0

def gene_Distribution(Distribution_file):
     # Distribution = pd.read_csv(Distribution_file, index_col=0, header=0)
    # 使用最小-最大缩放归一化方法对矩阵进行归一化对矩阵进行归一化
    # normalized_matrix = (Distribution_file - Distribution_file.min()) / (Distribution_file.max() - Distribution_file.min())
    Distribution_file = Distribution_file.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)
    # 绘制每个细胞系的分布图并保存为PNG文件
    for i in range(len(Distribution_file)):
        cell_line_name = Distribution_file.index[i]
        plt.hist(Distribution_file.iloc[i, :], bins=20)
        plt.title(str(cell_line_name) + ' Distribution')
        plt.xlabel('Normalized Scores')
        plt.ylabel('Frequency')
        plt.savefig('./result'+ str(cell_line_name) + '_distribution.png')
        plt.close()
    return 0

def get_top10gene(file):
    # matrix = pd.read_csv(file, index_col=0)

    # 要选取的三个细胞系
    cell_lines = [0,1,2,3]

    # 选取每个细胞系基因重要性得分前百分之10的基因
    selected_genes = {}
    selected_genes1 = {}
    gene_import = np.sum(file, axis=0)
    sorted_genes1 = gene_import.sort_values(ascending=False)
    series_head = sorted_genes1.head(20)
    print(series_head)
    for cell_line in cell_lines:
        sorted_genes = file.loc[cell_line].sort_values(ascending=False)
        selected_genes[cell_line] = sorted_genes.iloc[:int(0.1 * len(sorted_genes))]

    # 将结果保存为CSV文件
    for cell_line in selected_genes:
        selected_genes[cell_line].reset_index().to_csv('./result' + '_top10pct_genes.csv', columns=['index'],
                                                       index=False)
    return 0

if __name__ == '__main__':
    # 设置读取文件路径
    # args = opts.parse_args()
    # dataset_name= args.dataset_name
    #atten_get = np.zeros((128,2, 4,651))
    df_cell = pd.read_csv('./datasets/bindingdb/ONEIL-COSMIC/cell line_gene_expression.csv', index_col=0)
    columns = df_cell.columns.tolist()
    x = np.load('./result/arr_index.npy')
    for i in range(x.shape[0]):
        atten_get_head = np.zeros((1, 1, 4, 651))
        for j in range(x.shape[1]):
            atten_get_head = atten_get_head + x[i, j, :, :]
            #atten_get = atten_get.reshape(160, 651)
            #arr_1 = np.sum(atten_get[:150, :], axis=0)
            #arr_2 = np.sum(atten_get[150:, :], axis=0)
            #result = np.vstack([arr_1, arr_2])
        atten_get = atten_get_head[0,0,:,:]
        df = pd.DataFrame(atten_get)
            # print(x)
    #visual_cline = cline_edge_file = './result/attention-test.csv'
    # df_cell = pd.read_csv(visual_cline)
    #df_visual_cline = pd.read_csv(visual_cline,index_col=None, header=None)
    #columns = df_cell.columns.tolist()
        df.columns = columns
        if i in (7,16,20,38,49,66):
            Distribution_file = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)
            get_top10gene(Distribution_file)
            visual(Distribution_file)
            gene_import1 = np.sum(df, axis=0)
        #gene_import1 = gene_import1.reshape(651)
            sorted_genes2 = gene_import1.sort_values(ascending=False)
            series_head1 = sorted_genes2.head(20)
            #index = ['VINORELBINE', 'LAPATINIB']
            # df.index = index

           # gene_Distribution(Distribution_file)
            # get_top10gene(Distribution_file)