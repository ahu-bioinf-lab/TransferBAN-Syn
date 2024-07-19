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
        #all_data_path = os.path.join(dataFolder, 'drug-synergy-new.csv')
        # all_data_path = os.path.join(dataFolder, 'drug_synergy.csv')
        # all_data = pd.read_csv(all_data_path)
        # 划分数据集
        #kf = KFold(n_splits=5, shuffle=True, random_state=42)
        # df_drug = pd.read_csv('./datasets/bindingdb/NEW-echi-data/drug-smiles-new.csv')
        # df_drug.set_index('drug', inplace=True)
        # df_cell = pd.read_csv('./datasets/bindingdb/NEW-echi-data/cell-line-new.csv')
        #包虫病数据集上
        #修改1
        #所有寄生虫病上运行时
        all_data_path = os.path.join(dataFolder, 'DDS-name.csv')
        # all_data = pd.read_csv(all_data_path)
        #去除包虫病数据集
        all_data_path = os.path.join(dataFolder, 'source-dds-pros.csv')
        # all_data_path = os.path.join(dataFolder, 'target-dds-130.csv')
        all_data = pd.read_csv(all_data_path)
        # 划分数据集
        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        df_drug = pd.read_csv('./datasets/bindingdb/large-data/drug-smiles-new1.csv')
        df_drug.set_index('drug', inplace=True)
        df_cell_path = pd.read_csv('./datasets/bindingdb/large-data/pathway-large.csv')
        df_cell_sim = pd.read_csv('./datasets/bindingdb/large-data/disease-similar-1.csv')

        #large dataset
        #三行
        '''oneil
        df_cell = pd.read_csv('./datasets/bindingdb/ONEIL-COSMIC/cell line_gene_expression.csv', index_col=None)'''
        #ENEIL'的数据集
        #df_drug = pd.read_csv('./datasets/bindingdb/ONEIL-COSMIC/drug_smiles.csv')
        #df_drug.set_index('pubchemid',inplace=True)
        #column_values = df_cell['Pathway'].tolist()
        #dict_from_column = {i: value for i, value in enumerate(column_values)}


        # 输出创建的字典
        #print(dict_from_column)
        # cell_features = []
        # with open("./datasets/bindingdb/NEW-echi-data/cell-feature.csv", mode="r", newline="") as csv_file:
        #     csv_reader = csv.reader(csv_file)  # 使用csv.reader读取csvfile中的文件
        #     for row in csv_reader:
        #         cell_features.append(row)
        # cell_features = np.array(cell_features)
        # cell_features = pd.DataFrame(cell_features)
        # cell_features.set_index('Pathway', inplace=True)
        # print('cell_features', cell_features)
        #df_cell.set_index('Pathway', inplace=True)
        # cell_gene = np.load('./datasets/ALMANAC-COSMIC/cline_gene_ALMANAC.npy')
        #drug_feat = np.load('./drug/data/ALMANAC_drug_smiles_deal.npy')
        for fold_num, (train_index, validate_index) in enumerate(kf.split(all_data)):
            # 获取训练集和验证集
            train_data = all_data.iloc[train_index]
            validate_data = all_data.iloc[validate_index]
            # 将训练集和验证集保存到文件
            # train_data.to_csv(f'./datasets/bindingdb/ONEIL-COSMIC/train_data{fold_num}.csv', index=False)
            # val.to_csv(f'./datasets/bindingdb/ONEIL-COSMIC/val{fold_num}.csv', index=False)
            # #validate_data.to_csv(f'./datasets/ALMANAC-COSMIC/validate_data{fold_num}.csv', index=False)
            # train_data_path = os.path.join(dataFolder, f'train_data{fold_num}.csv')
            # val_data_path = os.path.join(dataFolder, f'val{fold_num}.csv')
            #修改2
            # train_data.to_csv(f'./datasets/bindingdb/large-data/pros-all-train_data{fold_num}.csv', index=False)
            # validate_data.to_csv(f'./datasets/bindingdb/large-data/pros-all-validate{fold_num}.csv', index=False)
            # 不进行迁移学习

            # train_data.to_csv(f'./datasets/bindingdb/large-data/target-only-train_data{fold_num}.csv', index=False)
            # validate_data.to_csv(f'./datasets/bindingdb/large-data/target_only-validate{fold_num}.csv', index=False)
           #  迁移学习--其他寄生虫病源域
           #  train_data.to_csv(f'./datasets/biosnap/cluster/sourece_train_data{fold_num}.csv', index=False)
           #  validate_data.to_csv(f'./datasets/biosnap/cluster/sourece_validate{fold_num}.csv', index=False)

            train_data.to_csv(f'./datasets/bindingdb/large-data/sourece-only-train_data{fold_num}.csv', index=False)
            validate_data.to_csv(f'./datasets/bindingdb/large-data/sourece-only-validate{fold_num}.csv', index=False)
            train_data_path = os.path.join(dataFolder, f'sourece-only-train_data{fold_num}.csv')
            val_data_path = os.path.join(dataFolder,  f'sourece-only-validate{fold_num}.csv')
            
            # train_data.to_csv(f'./datasets/biosnap/cluster/fine-tuning_target_train_data{fold_num}.csv', index=False)
            # validate_data.to_csv(f'./datasets/biosnap/cluster/fine-tuning_target_validate{fold_num}.csv', index=False)
            # train_data_path = os.path.join(dataFolder, f'fine-tuning_target_train_data{fold_num}.csv')
            # val_data_path = os.path.join(dataFolder, f'fine-tuning_target_validate{fold_num}.csv')


            # train_data.to_csv(f'./datasets/biosnap/cluster/train_data{fold_num}.csv', index=False)
            # validate_data.to_csv(f'./datasets/biosnap/cluster/validate{fold_num}.csv', index=False)
          #  fold_num  = 3
          #   train_data_path = os.path.join(dataFolder, f'target-only-train_data{fold_num}.csv')
          #   val_data_path = os.path.join(dataFolder, f'target_only-validate{fold_num}.csv')
            # test_data_path = os.path.join(dataFolder, 'validate_fold1.csv')
        # train_path = os.path.join(dataFolder, 'train.csv')
        # val_path = os.path.join(dataFolder, "val.csv")

            # train_data_path = os.path.join(dataFolder, f'source-dds.csv')
            # val_data_path = os.path.join(dataFolder, f'target-dds-130.csv')



            #修改3
            test_path = os.path.join(dataFolder,   f'dependence-drug-comb-testxlsx.csv')
            df_train = pd.read_csv(train_data_path)
            df_val = pd.read_csv(val_data_path)
            #修改4
            df_test = pd.read_csv(test_path)
            train_dataset = DTIDataset(df_train.index.values, df_train,df_drug,df_cell_path,df_cell_sim)
            #train_dataset = DTIDataset(df_train.index.values, df_train)
        # print()
            val_dataset = DTIDataset(df_val.index.values, df_val,df_drug,df_cell_path,df_cell_sim)
            test_dataset = DTIDataset(df_test.index.values, df_test,df_drug,df_cell_path,df_cell_sim)
            
            # train_dataset = DTIDataset(df_train.index.values, df_train,df_drug,df_cell)
            # val_dataset = DTIDataset(df_val.index.values, df_val,df_drug,df_cell)
            #test_dataset = DTIDataset(df_test.index.values, df_test)

            params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
                'drop_last': True, 'collate_fn': graph_collate_func}
            #params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              #'drop_last': True}
            if not cfg.DA.USE:
                training_generator = DataLoader(train_dataset, **params)
                params['shuffle'] = True
                params['drop_last'] = True
                if not cfg.DA.TASK:
                    val_generator = DataLoader(val_dataset, **params)
                    #修改5
                    test_generator = DataLoader(test_dataset, **params)
            model = DrugBAN(**cfg).to(device)
            # for name, param in model.named_parameters():
            #     if "mlp_classifier.fc4" in name or "mlp_classifier.fc5" in name:
            #         param.requires_grad_(True)
            #     else:
            #         param.requires_grad_(False)

            # for name, param in model.named_parameters():
            #     param.requires_grad_(False)

            # model.mlp_classifier =  nn.Linear(384, binary = 1)
            # for name, param in model.named_parameters():
            #     if "mlp_classifier" in name:
            #         param.requires_grad_(True)
            #     else:
            #         param.requires_grad_(False)
                    # param.append(param)
            #         print("\t", name)
            #检查是否有任何参数被冻结

            # if all(param.requires_grad is False for param in model.parameters()):
            #     raise ValueError("所有参数都已被冻结。优化器收到了一个空的参数列表。")
            # model.load_state_dict(torch.load("./result/best_model_epoch_82.pth"))
            # 创建 Adam 优化器，但只更新特定的参数
            # opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.SOLVER.LR)

            opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
            torch.backends.cudnn.benchmark = True
            scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5, verbose=True)
            #修改6
            trainer = Trainer(model, scheduler,opt, device, training_generator, val_generator,test_generator,fold_num,opt_da=None,
                          discriminator=None,
                          experiment=experiment, **cfg)

            # trainer = Trainer_1(model, opt, device, training_generator, val_generator, fold_num,
            #                   opt_da=None,
            #                   discriminator=None,
            #                   experiment=experiment, **cfg)
            result = trainer.train()

            with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
                wf.write(str(model))
                print()
                print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")
        return result
    else:
        train_source_path = os.path.join(dataFolder, 'source-dds.csv')
        df_train_source = pd.read_csv(train_source_path)
        target_data_path = os.path.join(dataFolder, 'target-dds.csv')
        target_data = pd.read_csv(target_data_path)
        # 划分数据集
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        df_drug = pd.read_csv('./datasets/biosnap/cluster/drug-smiles-new1.csv')
        df_drug.set_index('drug', inplace=True)
        df_cell_path = pd.read_csv('./datasets/biosnap/cluster/pathway-large.csv')
        df_cell_sim = pd.read_csv('./datasets/biosnap/cluster/disease-similar-1.csv')

        for fold_num, (train_index, validate_index) in enumerate(kf.split(target_data)):
            # 获取训练集和验证集
            train_target_path = target_data.iloc[train_index]
            test_target_path = target_data.iloc[validate_index]
            # 将训练集和验证集保存到文件
            # train_data.to_csv(f'./datasets/bindingdb/ONEIL-COSMIC/train_data{fold_num}.csv', index=False)
            # val.to_csv(f'./datasets/bindingdb/ONEIL-COSMIC/val{fold_num}.csv', index=False)
            # #validate_data.to_csv(f'./datasets/ALMANAC-COSMIC/validate_data{fold_num}.csv', index=False)
            # train_data_path = os.path.join(dataFolder, f'train_data{fold_num}.csv')
            # val_data_path = os.path.join(dataFolder, f'val{fold_num}.csv')
            # 修改2
            train_target_path.to_csv(f'./datasets/biosnap/cluster/target_train_data{fold_num}.csv', index=False)
            test_target_path.to_csv(f'./datasets/biosnap/cluster/target_validate{fold_num}.csv', index=False)
            train_data_path = os.path.join(dataFolder, f'target_train_data3.csv')
            val_data_path = os.path.join(dataFolder, f'target_validate3.csv')

            #test_path = os.path.join(dataFolder, f'dependence-drug-comb-testxlsx.csv')
            df_train_target = pd.read_csv(train_data_path)
            df_test_target = pd.read_csv(val_data_path)
            # 修改4
            #df_test = pd.read_csv(test_path)
            train_source_dataset = DTIDataset(df_train_source.index.values, df_train_source, df_drug, df_cell_path, df_cell_sim)
            # train_dataset = DTIDataset(df_train.index.values, df_train)
            # print()
            train_target_dataset = DTIDataset(df_train_target.index.values, df_train_target, df_drug, df_cell_path, df_cell_sim)
            test_target_dataset = DTIDataset(df_test_target.index.values, df_test_target, df_drug, df_cell_path, df_cell_sim)
            params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
                      'drop_last': True, 'collate_fn': graph_collate_func}

            source_generator = DataLoader(train_source_dataset, **params)
            target_generator = DataLoader(train_target_dataset, **params)
            n_batches = max(len(source_generator), len(target_generator))
            multi_generator = MultiDataLoader(dataloaders=[source_generator, target_generator], n_batches=n_batches)
            params['shuffle'] = True
            params['drop_last'] = False
            val_generator = DataLoader(test_target_dataset, **params)
            test_generator = DataLoader(test_target_dataset, **params)

            model = DrugBAN(**cfg).to(device)

            if cfg.DA.USE:
                if cfg["DA"]["RANDOM_LAYER"]:
                    domain_dmm = Discriminator(input_size=cfg["DA"]["RANDOM_DIM"], n_class=cfg["DECODER"]["BINARY"]).to(
                        device)
                else:
                    domain_dmm = Discriminator(input_size=cfg["DECODER"]["IN_DIM"] * cfg["DECODER"]["BINARY"],
                                               n_class=cfg["DECODER"]["BINARY"]).to(device)
                # params = list(model.parameters()) + list(domain_dmm.parameters())
                opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
                opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)
            else:
                opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

            torch.backends.cudnn.benchmark = True

            trainer = Trainer(model, opt, device, multi_generator, val_generator, test_generator, fold_num,opt_da=opt_da,
                                  discriminator=domain_dmm,
                                  experiment=experiment, **cfg)
            result = trainer.train()

            with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
                wf.write(str(model))

            print()
            print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

            return result

    # return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
