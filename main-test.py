comet_support = True
try:
    from comet_ml import Experiment
except ImportError as e:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False
from models import DrugBAN
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
# from ONEIL_dataloader_printfinger_and_discriptor import DTIDataset, MultiDataLoader
from dataloader_molegraph import DTIDataset, MultiDataLoader
from torch.utils.data import DataLoader
from trainer import Trainer
from train_cancer import Trainer_1
from domain_adaptator import Discriminator
import torch
import argparse
import warnings, os
import pandas as pd
from sklearn.model_selection import KFold
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import csv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
parser.add_argument('--cfg', required=True, help="path to config file", type=str)
parser.add_argument('--data', required=True, type=str, metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random', 'cold', 'cluster','ONEIL-COSMIC','ALMANAC-COSMIC','NEW-echi-data','large-data','echinococcosis-data','AAA'])
args = parser.parse_args()

device = torch.device('cpu')


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]
    mkdir(cfg.RESULT.OUTPUT_DIR)
    experiment = None
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1, 2"
    torch.cuda.set_device(0)
    dataFolder = f'./datasets/{args.data}' #oneil数据集
    #dataFolder = f'./datasets'
    dataFolder = os.path.join(dataFolder, str(args.split))
    # model = DrugBAN().to(device)
    # pretrained_model_path = './pretrained_model/best_model_epoch_61.pth'
    # # print(torch.load(pretrained_model_path).keys())  # 替换为您的模型实现
    # model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    # model.eval()
    if not cfg.DA.TASK:
        df_drug = pd.read_csv('./datasets/bindingdb/large-data/drug-smiles-new1.csv')
        df_drug.set_index('drug', inplace=True)
        df_cell_path = pd.read_csv('./datasets/bindingdb/large-data/pathway-large.csv')
        df_cell_sim = pd.read_csv('./datasets/bindingdb/large-data/disease-similar-1.csv')
        params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': False, 'num_workers': cfg.SOLVER.NUM_WORKERS,
                  'drop_last': True, 'collate_fn': graph_collate_func}

        # df_test = pd.read_csv("./datasets/bindingdb/large-data/dependence-drug-comb-testxlsx.csv")
        # test_dataset = DTIDataset(df_test.index.values, df_test, df_drug, df_cell_path, df_cell_sim)
        # test_generator = DataLoader(test_dataset, **params)
        model = DrugBAN(**cfg).to(device)

        # 步骤 1: 加载模型参数
        model.load_state_dict(torch.load("./result/best_model_epoch_61_test_no_echi_positive.pth"))
        model.eval()

        # 步骤 2: 加载独立验证集数据
        df_test = pd.read_csv("./datasets/bindingdb/large-data/dependence-drug-comb-test-all.csv") # 加载独立验证集文件

        # 步骤 3: 准备数据集和数据加载器
        test_dataset = DTIDataset(df_test.index.values, df_test, df_drug, df_cell_path, df_cell_sim)
        test_generator = DataLoader(test_dataset, **params)

        # 步骤 4: 预测并保存分类得分
        all_scores = []
        with torch.no_grad():
            model.eval()
            for i, (v_d1,v_d2,d1_print,d2_print, v_p, labels,index) in enumerate(test_generator):
                v_p = [torch.tensor(arr, dtype=torch.float32) for arr in v_p]
                v_p = torch.cat(v_p, dim=0)
                v_d1, v_d2, v_p, labels = v_d1.to(device), v_d2.to(device), v_p.to(
                    device), labels.float().to(device)  # 如果使用 GPU，需要将数据移动到 GPU 上
                v_d, v_pt, score =model(v_d1, v_d2, v_p, mode="eval")  # 如果是二分类任务，应用 sigmoid 函数
                m = nn.Sigmoid()
                n = m(score)
                all_scores.append(n.cpu().numpy().tolist())

        # 将所有得分拼接成一个列表
        all_scores_flat = [item for sublist in all_scores for item in sublist]
        df_test = df_test.iloc[:len(all_scores_flat)]
        # 将得分添加到 DataFrame 中
        df_test['score'] = all_scores_flat
        df_test.to_csv("./datasets/bindingdb/large-data/dependence-drug-comb-test-all-with-scores-sigmoid-34287-CE.csv", index=False)  # 保存带有分类得分的文件



if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
