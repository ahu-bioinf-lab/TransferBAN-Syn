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
from domain_adaptator import Discriminator
import torch
import argparse
import warnings, os
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
parser.add_argument('--cfg', required=True, help="path to config file", type=str)
parser.add_argument('--data', required=True, type=str, metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random', 'cold', 'cluster','ONEIL-COSMIC','ALMANAC-COSMIC'])
args = parser.parse_args()


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

    if not cfg.DA.TASK:
        all_data_path = os.path.join(dataFolder, 'drug_synergy.csv')
        all_data = pd.read_csv(all_data_path)
        # 划分数据集
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        df_drug = pd.read_csv('./datasets/bindingdb/ONEIL-COSMIC/drug_smiles.csv') #ENEIL'的数据集
        df_drug.set_index('pubchemid', inplace=True) #ENEIL'的数据集
        df_cell = pd.read_csv('./datasets/bindingdb/ONEIL-COSMIC/cell line_gene_expression.csv', index_col=0) #ENEIL'的数据集
        cell_gene = np.load('./datasets/bindingdb/ONEIL-COSMIC/cline_gene.npy') #ENEIL'的数据集
        drug_feat = np.load('./drug/data/drug_feat.npy')  # ENEIL'的数据集
        # df_drug = pd.read_csv('./datasets/ALMANAC-COSMIC/drug_smiles.csv')
        # df_drug.reset_index(inplace=True)
        # df_cell = pd.read_csv('./datasets/ALMANAC-COSMIC/cell line_gene_expression.csv', index_col=0)
        # cell_gene = np.load('./datasets/ALMANAC-COSMIC/cline_gene_ALMANAC.npy')
        #drug_feat = np.load('./drug/data/ALMANAC_drug_smiles_deal.npy')
        file_path = './drug/data/drug2id.tsv'
        drug2id = pd.read_csv(file_path, delimiter='\t')
        drug2id.set_index('drug', inplace=True)
        for fold_num, (train_index, validate_index) in enumerate(kf.split(all_data)):
            # 获取训练集和验证集
            train_data = all_data.iloc[train_index]
            validate_data = all_data.iloc[validate_index]
            # 将训练集和验证集保存到文件
            # train_data.to_csv(f'./datasets/ALMANAC-COSMIC/train_data{fold_num}.csv', index=False)
            # validate_data.to_csv(f'./datasets/ALMANAC-COSMIC/validate_fold{fold_num}.csv', index=False)
            train_data.to_csv(f'./datasets/bindingdb/ONEIL-COSMIC/train_data{fold_num}.csv', index=False)
            validate_data.to_csv(f'./datasets/bindingdb/ONEIL-COSMIC/validate_fold{fold_num}.csv', index=False)
            train_data_path = os.path.join(dataFolder, f'train_data{fold_num}.csv')
            val_data_path = os.path.join(dataFolder,   f'validate_fold{fold_num}.csv')
        # test_data_path = os.path.join(dataFolder, 'validate_fold1.csv')
        # train_path = os.path.join(dataFolder, 'train.csv')
        # val_path = os.path.join(dataFolder, "val.csv")
        # test_path = os.path.join(dataFolder, "test.csv")
            df_train = pd.read_csv(train_data_path)
            df_val = pd.read_csv(val_data_path)
        # df_test = pd.read_csv(test_data_path)
        #train_dataset = DTIDataset(df_train.index.values, df_train)
            train_dataset = DTIDataset(df_train.index.values, df_train,df_drug,df_cell,cell_gene,drug_feat)
        # print()
            val_dataset = DTIDataset(df_val.index.values, df_val,df_drug,df_cell,cell_gene,drug_feat)
        # test_dataset = DTIDataset(df_test.index.values, df_test,df_drug,df_cell,cell_gene)
        # train_dataset = DTIDataset(df_train.index.values, df_train)
        # val_dataset = DTIDataset(df_val.index.values, df_val)
        # test_dataset = DTIDataset(df_test.index.values, df_test)

    # params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
             # 'drop_last': True, 'collate_fn': graph_collate_func}
            params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True}
            if not cfg.DA.USE:
                training_generator = DataLoader(train_dataset, **params)
                params['shuffle'] = False
                params['drop_last'] = True
                if not cfg.DA.TASK:
                    val_generator = DataLoader(val_dataset, **params)
            # test_generator = DataLoader(test_dataset, **params)

            model = DrugBAN(**cfg).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

            torch.backends.cudnn.benchmark = True

            trainer = Trainer(model, opt, device, training_generator, val_generator, opt_da=None,
                          discriminator=None,
                          experiment=experiment, **cfg)
            result = trainer.train()

            with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
                wf.write(str(model))
            return result
    print()
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    # return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
