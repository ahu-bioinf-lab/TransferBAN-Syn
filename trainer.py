import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score,accuracy_score,cohen_kappa_score,recall_score
from models import binary_cross_entropy, cross_entropy_logits, entropy_logits, RandomLayer, DrugBAN
from prettytable import PrettyTable
from domain_adaptator import ReverseLayerF
from tqdm import tqdm
import pandas as pd

class Trainer(object):
    def __init__(self, model,scheduler, optim, device, train_dataloader, val_dataloader, test_dataloader,fold_num,opt_da=None, discriminator=None,
                 experiment=None, alpha=1, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.is_da = config["DA"]["USE"]
        self.alpha = alpha
        self.sch=scheduler
        self.n_class = config["DECODER"]["BINARY"]
        if opt_da:
            self.optim_da = opt_da
        if self.is_da:
            self.da_method = config["DA"]["METHOD"]
            self.domain_dmm = discriminator
            if config["DA"]["RANDOM_LAYER"] and not config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = nn.Linear(in_features=config["DECODER"]["IN_DIM"]*self.n_class, out_features=config["DA"]
                ["RANDOM_DIM"], bias=False).to(self.device)
                torch.nn.init.normal_(self.random_layer.weight, mean=0, std=1)
                for param in self.random_layer.parameters():
                    param.requires_grad = False
            elif config["DA"]["RANDOM_LAYER"] and config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = RandomLayer([config["DECODER"]["IN_DIM"], self.n_class], config["DA"]["RANDOM_DIM"])
                if torch.cuda.is_available():
                    self.random_layer.cuda()
            else:
                self.random_layer = False
        self.da_init_epoch = config["DA"]["INIT_EPOCH"]
        self.init_lamb_da = config["DA"]["LAMB_DA"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.use_da_entropy = config["DA"]["USE_ENTROPY"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.experiment = experiment

        self.best_epoch = None
        self.best_auroc = 0
        self.epoch_time = fold_num
        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss"]
        valid_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
        "Threshold", "Test_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Threshold", "Test_loss"]
        if not self.is_da:
            train_metric_header = ["# Epoch", "Train_loss"]
        else:
            train_metric_header = ["# Epoch", "Train_loss", "Model_loss", "epoch_lamb_da", "da_loss"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

        self.original_random = config["DA"]["ORIGINAL_RANDOM"]

    def da_lambda_decay(self):
        delta_epoch = self.current_epoch - self.da_init_epoch
        non_init_epoch = self.epochs - self.da_init_epoch
        p = (self.current_epoch + delta_epoch * self.nb_training) / (
                non_init_epoch * self.nb_training
        )
        grow_fact = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        return self.init_lamb_da * grow_fact

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            if not self.is_da:
                train_loss = self.train_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
                if self.experiment:
                    self.experiment.log_metric("train_epoch model loss", train_loss, epoch=self.current_epoch)
            else:
                train_loss, model_loss, da_loss, epoch_lamb = self.train_da_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss, model_loss,
                                                                                        epoch_lamb, da_loss]))
                self.train_model_loss_epoch.append(model_loss)
                self.train_da_loss_epoch.append(da_loss)
                if self.experiment:
                    self.experiment.log_metric("train_epoch total loss", train_loss, epoch=self.current_epoch)
                    self.experiment.log_metric("train_epoch model loss", model_loss, epoch=self.current_epoch)
                    if self.current_epoch >= self.da_init_epoch:
                        self.experiment.log_metric("train_epoch da loss", da_loss, epoch=self.current_epoch)
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            self.sch.step(train_loss)  #动态调整学习率
           #  auroc, auprc, val_loss = self.test(dataloader="val")
            auroc1,auprc1, f11, sensitivity1, specificity1, accuracy1, val_loss1, thred_optim1, precision1,index = self.test(dataloader="val")
            if self.experiment:
                self.experiment.log_metric("valid_epoch model loss", val_loss1, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auroc", auroc1, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auprc", auprc1, epoch=self.current_epoch)
            # val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc1, auprc1, val_loss1]))
            val_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc1, auprc1,f11, sensitivity1, precision1,
                                                                     accuracy1, thred_optim1, val_loss1]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss1)
            self.val_auroc_epoch.append(auroc1)
            #3#self.
            if auroc1 >= self.best_auroc:
                #torch.save(self.model.state_dict(), './datasets/bindingdb/NEW-echi-data/drug-smiles-new.pth')
                #self.best_model = copy.deepcopy(self.model)
                # 假设你的模型是model
                self.best_model = self.model
                model_state_dict = self.model.state_dict()
                # 创建一个新的模型
                self.best_model.load_state_dict(copy.deepcopy(model_state_dict))  # 请根据你的模型类进行调整

                self.best_auroc = auroc1
                self.best_epoch = self.current_epoch
                # att = att.numpy()
                #np.save(f'./result/arr_index.npy', att)
                print('best_auroc:------',auroc1)
                # print('index!!!!!!!', index)
                # atten_get = att[0, 0, :, :]
                # atten_get = atten_get.reshape(160, 651)
                # arr_1 = np.sum(atten_get[:150, :], axis=0)
                # arr_2 = np.sum(atten_get[150:, :], axis=0)
                # result = np.vstack([arr_1, arr_2])
                # df = pd.DataFrame(result)
                # df.to_csv('./result/attention-test.csv', index=False, header=False)
            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss1), " AUROC "
                  + str(auroc1) + " AUPRC " + str(auprc1)+"accuracy"+str(accuracy1)+"precision"+str(precision1))
        #auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision,att1 = self.test(dataloader="test")

        # test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity,
        #                                                                     accuracy, thred_optim, test_loss]))
        # self.test_table.add_row(test_lst)
        # print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " AUROC "
        #       + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity " +
        #       str(specificity) + " Accuracy " + str(accuracy) + " Thred_optim " + str(thred_optim))
        self.test_metrics["auroc"] = auroc1
        self.test_metrics["auprc"] = auprc1
        self.test_metrics["test_loss"] = val_loss1
        self.test_metrics["sensitivity"] = sensitivity1
        self.test_metrics["Precision_1"] = precision1
        self.test_metrics["accuracy"] = accuracy1
        self.test_metrics["thred_optim"] = thred_optim1
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f11
        self.test_metrics["Precision"] = precision1
        self.save_result()
        if self.experiment:
            self.experiment.log_metric("valid_best_auroc", self.best_auroc)
            self.experiment.log_metric("valid_best_epoch", self.best_epoch)
            self.experiment.log_metric("test_auroc", self.test_metrics["auroc"])
            self.experiment.log_metric("test_auprc", self.test_metrics["auprc"])
            self.experiment.log_metric("test_sensitivity", self.test_metrics["sensitivity"])
            self.experiment.log_metric("test_specificity", self.test_metrics["specificity"])
            self.experiment.log_metric("test_accuracy", self.test_metrics["accuracy"])
            self.experiment.log_metric("test_threshold", self.test_metrics["thred_optim"])
            self.experiment.log_metric("test_f1", self.test_metrics["F1"])
            self.experiment.log_metric("test_precision", self.test_metrics["Precision"])
        #v_d, v_pt, score, at = self.test(dataloader="test")
        # v_d, v_pt, score = self.test(dataloader="test")
        print("!!!!!")
        # print(index,score)
        return self.test_metrics

    def save_result(self):
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        if self.is_da:
            state["train_model_loss"] = self.train_model_loss_epoch
            state["train_da_loss"] = self.train_da_loss_epoch
            state["da_init_epoch"] = self.da_init_epoch
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        # val_prettytable_file = os.path.join(self.output_dir, f"without-transfer-echi_only_valid_markdowntable_{self.epoch_time}.txt")
        # test_prettytable_file = os.path.join(self.output_dir, f"without-transfer-echi_only-test_markdowntable_{self.epoch_time}.txt")
        # train_prettytable_file = os.path.join(self.output_dir, f"without-transfer-echi_only-test-target_train_markdowntable_{self.epoch_time}.txt")
        # val_prettytable_file = os.path.join(self.output_dir, f"pros_target_valid_markdowntable_{self.epoch_time}.txt")
        # test_prettytable_file = os.path.join(self.output_dir, f"pros_markdowntable_{self.epoch_time}.txt")
        # train_prettytable_file = os.path.join(self.output_dir, f"pros_target_train_markdowntable_{self.epoch_time}.txt")
        val_prettytable_file = os.path.join(self.output_dir,
                                            f"learning-rat5e3-echi_only_valid_markdowntable_{self.epoch_time}.txt")
        test_prettytable_file = os.path.join(self.output_dir, f"learning-rat5e3-without-transfer-echi_only-test_markdowntable_{self.epoch_time}.txt")
        train_prettytable_file = os.path.join(self.output_dir, f"learning-rat5e3-without-transfer-echi_only-test-target_train_markdowntable_{self.epoch_time}.txt")

        self.epoch_time = self.epoch_time + 1
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def _compute_entropy_weights(self, logits):
        entropy = entropy_logits(logits)
        entropy = ReverseLayerF.apply(entropy, self.alpha)
        entropy_w = 1.0 + torch.exp(-entropy)
        return entropy_w

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (v_d1, v_d2,d1_print,d2_print,v_p, labels,index) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            # all_node_list = []
            # x_list = v_d.batch_num_nodes().tolist()
            # for index, item in enumerate(x_list):
            #     all_node_list.append([index]*item)
            # # print(all_node_list)
            # all_node_tensor = torch.cat([torch.tensor(sublist) for sublist in all_node_list])
            # all_node_tensor = torch.tensor(all_node_list)
            # all_node_tensor =all_node_tensor.flatten()
            # print(index)
            #v_p =np.array(v_p)
            v_p = [torch.tensor(arr, dtype=torch.float32) for arr in v_p]
            v_p = torch.cat(v_p, dim=0)
            #v_p = np.array(v_p)
            #v_p = torch.from_numpy(v_p)
            # v_p = torch.tensor(v_p, dtype=torch.float32)
            # v_p = torch.from_numpy(v_p)
            # v_p = v_p.astype(float)
            v_d1, v_d2,v_p, labels= v_d1.to(self.device), v_d2.to(self.device),v_p.to(self.device), labels.float().to(self.device)
            self.optim.zero_grad()
            v_d, v_p, f, score = self.model(v_d1,v_d2, v_p,mode="train")
            if self.n_class == 1:
                n, loss = binary_cross_entropy(score, labels)
            else:
                n, loss = cross_entropy_logits(score, labels)

            loss.backward()
            self.optim.step()
            # self.sche.step(val_loss)
            loss_epoch += loss.item()
            if self.experiment:
                self.experiment.log_metric("train_step model loss", loss.item(), step=self.step)
        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch

    def train_da_epoch(self):
        self.model.train()
        total_loss_epoch = 0
        model_loss_epoch = 0
        da_loss_epoch = 0
        epoch_lamb_da = 0
        if self.current_epoch >= self.da_init_epoch:
            # epoch_lamb_da = self.da_lambda_decay()
            epoch_lamb_da = 1
            if self.experiment:
                self.experiment.log_metric("DA loss lambda", epoch_lamb_da, epoch=self.current_epoch)
        num_batches = len(self.train_dataloader)
        for i, (batch_s, batch_t) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            v_d1, v_d2 = batch_s[0].to(self.device), batch_s[1].to(self.device)
            v_p = batch_s[2]
            v_p = [torch.tensor(arr, dtype=torch.float32) for arr in v_p]
            v_p = torch.cat(v_p, dim=0)
            v_p = v_p.to(self.device)
            labels = batch_s[3].float().to(self.device)
            v_d1_t, v_d2_t= batch_t[0].to(self.device), batch_t[1].to(self.device)
            v_p_t = batch_t[2]
            v_p_t = [torch.tensor(arr, dtype=torch.float32) for arr in v_p_t]
            v_p_t = torch.cat(v_p_t, dim=0)
            v_p_t = v_p_t.to(self.device)
            self.optim.zero_grad()
            self.optim_da.zero_grad()
            v_d, v_p, f, score = self.model(v_d1,v_d2, v_p,mode="train")
            if self.n_class == 1:
                n, model_loss = binary_cross_entropy(score, labels)
            else:
                n, model_loss = cross_entropy_logits(score, labels)
            if self.current_epoch >= self.da_init_epoch:
                v_d_t, v_p_t, f_t, t_score = self.model(v_d1_t, v_d2_t, v_p_t,mode = "train")
                if self.da_method == "CDAN":
                    reverse_f = ReverseLayerF.apply(f, self.alpha)
                    softmax_output = torch.nn.Softmax(dim=1)(score)
                    softmax_output = softmax_output.detach()
                    # reverse_output = ReverseLayerF.apply(softmax_output, self.alpha)
                    if self.original_random:
                        random_out = self.random_layer.forward([reverse_f, softmax_output])
                        adv_output_src_score = self.domain_dmm(random_out.view(-1, random_out.size(1)))
                    else:
                        feature = torch.bmm(softmax_output.unsqueeze(2), reverse_f.unsqueeze(1))
                        feature = feature.view(-1, softmax_output.size(1) * reverse_f.size(1))
                        if self.random_layer:
                            random_out = self.random_layer.forward(feature)
                            adv_output_src_score = self.domain_dmm(random_out)
                        else:
                            adv_output_src_score = self.domain_dmm(feature)

                    reverse_f_t = ReverseLayerF.apply(f_t, self.alpha)
                    softmax_output_t = torch.nn.Softmax(dim=1)(t_score)
                    softmax_output_t = softmax_output_t.detach()
                    # reverse_output_t = ReverseLayerF.apply(softmax_output_t, self.alpha)
                    if self.original_random:
                        random_out_t = self.random_layer.forward([reverse_f_t, softmax_output_t])
                        adv_output_tgt_score = self.domain_dmm(random_out_t.view(-1, random_out_t.size(1)))
                    else:
                        feature_t = torch.bmm(softmax_output_t.unsqueeze(2), reverse_f_t.unsqueeze(1))
                        feature_t = feature_t.view(-1, softmax_output_t.size(1) * reverse_f_t.size(1))
                        if self.random_layer:
                            random_out_t = self.random_layer.forward(feature_t)
                            adv_output_tgt_score = self.domain_dmm(random_out_t)
                        else:
                            adv_output_tgt_score = self.domain_dmm(feature_t)

                    if self.use_da_entropy:
                        entropy_src = self._compute_entropy_weights(score)
                        entropy_tgt = self._compute_entropy_weights(t_score)
                        src_weight = entropy_src / torch.sum(entropy_src)
                        tgt_weight = entropy_tgt / torch.sum(entropy_tgt)
                    else:
                        src_weight = None
                        tgt_weight = None
                    # 使用 Sigmoid 函数替代 BCEWithLogitsLoss
                    n_src, loss_cdan_src = cross_entropy_logits(adv_output_src_score, torch.zeros(self.batch_size).to(self.device),
                                                                src_weight)
                    n_tgt, loss_cdan_tgt = cross_entropy_logits(adv_output_tgt_score, torch.ones(self.batch_size).to(self.device),
                                                                tgt_weight)
                    da_loss = loss_cdan_src + loss_cdan_tgt
                    # criterion = torch.nn.BCEWithLogitsLoss()
                    # # 将概率值转换为二元类别标签
                    # loss_cdan_tgt = criterion(adv_output_src_score.view(-1),
                    #                           torch.ones_like(adv_output_src_score).to(self.device), weight=src_weight )
                    # loss_cdan_src = criterion(adv_output_tgt_score.view(-1),
                    #                           torch.ones_like(adv_output_tgt_score).to(self.device), weight=tgt_weight)
                    da_loss = loss_cdan_src + loss_cdan_tgt
                else:
                    raise ValueError(f"The da method {self.da_method} is not supported")
                loss = model_loss + da_loss
            else:
                loss = model_loss
            loss.backward()
            self.optim.step()
            self.optim_da.step()
            total_loss_epoch += loss.item()
            model_loss_epoch += model_loss.item()
            if self.experiment:
                self.experiment.log_metric("train_step model loss", model_loss.item(), step=self.step)
                self.experiment.log_metric("train_step total loss", loss.item(), step=self.step)
            if self.current_epoch >= self.da_init_epoch:
                da_loss_epoch += da_loss.item()
                if self.experiment:
                    self.experiment.log_metric("train_step da loss", da_loss.item(), step=self.step)
        total_loss_epoch = total_loss_epoch / num_batches
        model_loss_epoch = model_loss_epoch / num_batches
        da_loss_epoch = da_loss_epoch / num_batches
        if self.current_epoch < self.da_init_epoch:
            print('Training at Epoch ' + str(self.current_epoch) + ' with model training loss ' + str(total_loss_epoch))
        else:
            print('Training at Epoch ' + str(self.current_epoch) + ' model training loss ' + str(model_loss_epoch)
                  + ", da loss " + str(da_loss_epoch) + ", total training loss " + str(total_loss_epoch) + ", DA lambda " +
                  str(epoch_lamb_da))
        return total_loss_epoch, model_loss_epoch, da_loss_epoch, epoch_lamb_da

    def test(self, dataloader="test"):#只有在测试集上才会跑到该函数
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
            index_values = []
            score_values = []
            # data = {'索引值': index_values, '分数值': score_values}
            for i, (v_d1,v_d2,d1_print,d2_print, v_p, labels,index) in enumerate(data_loader):
                # all_node_list = []
                # x_list = v_d.batch_num_nodes().tolist()
                # for index, item in enumerate(x_list):
                #     all_node_list.append([index] * item)
                #
                # all_node_tensor = torch.cat([torch.tensor(sublist) for sublist in all_node_list])
                # all_node_tensor = torch.tensor(all_node_list)
                # all_node_tensor =all_node_tensor.flatten()
                v_p = [torch.tensor(arr, dtype=torch.float32) for arr in v_p]
                v_p = torch.cat(v_p, dim=0)
                v_d1, v_d2,v_p, labels = v_d1.to(self.device),v_d2.to(self.device), v_p.to(self.device), labels.float().to(
                    self.device)
                #v_d, v_pt, score, at = self.best_model(v_d1, v_d2, v_p, mode="eval") BAN
                if dataloader == "val":
                    v_d, v_pt, score = self.best_model(v_d1, v_d2, v_p, mode="eval")
                elif dataloader == "test":
                    v_d, v_pt, score = self.best_model(v_d1, v_d2, v_p, mode="eval")
                #print(index,score)
                m = nn.Sigmoid()
                n = torch.squeeze(m(score), 1)
                score_values.append(n)
                #my_tensor = torch.tensor(index)
            df = pd.DataFrame()
            for i, tensor_data in enumerate(score_values):
                    # 将 Tensor 数据转换为 Python 列表
                data_as_list = tensor_data.squeeze().tolist()

                    # 将数据添加到 DataFrame 中，列名可以根据需要指定
                df[f'Data_{i + 1}'] = data_as_list

                # 指定要保存的 CSV 文件名
            csv_file = './datasets/bindingdb/large-data/AE_tensor_data.csv'
                # 使用 to_csv 方法将数据保存到 CSV 文件
            df.to_csv(csv_file, index=False)
                # index_values = index_values.append(my_tensor)
            #df = pd.DataFrame(data)
            #df.to_excel('./datasets/bindingdb/NEW-echi-data/score-get.xlsx', sheet_name='数据', index=False)
            return v_d, v_pt, score,
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for i, (v_d1,v_d2, v_d1_fingerprint,v_d2_fingerprint, v_p,labels,index) in enumerate(data_loader):
                # all_node_list = []
                # x_list = v_d.batch_num_nodes().tolist()
                # for index, item in enumerate(x_list):
                #     all_node_list.append([index] * item)
                #
                # all_node_tensor = torch.cat([torch.tensor(sublist) for sublist in all_node_list])
                # all_node_tensor = torch.tensor(all_node_list)
                # all_node_tensor =all_node_tensor.flatten()
                v_p = [torch.tensor(arr, dtype=torch.float32) for arr in v_p]
                v_p = torch.cat(v_p, dim=0)
                v_d1, v_d2,v_p, labels = v_d1.to(self.device),v_d2.to(self.device), v_p.to(self.device), labels.float().to(
                    self.device)
                # print(all_node_tensor.shape)
                if dataloader == "val":
                    # v_d, v_p, f, score = self.model(v_d1,v_d2, v_p,mode="eval" )
                    # v_d, v_p, score, att = self.model(v_d1,v_d2, v_p,mode="eval" )     BAN
                    v_d, v_p, score = self.model(v_d1,v_d2, v_p,mode="eval" )
                elif dataloader == "test":
                    v_d, v_pt, score = self.best_model(v_d1, v_d2, v_p, mode="eval")
                    # att = att.cpu()
                    # att = att.numpy()
                    # atten_get = att[0,0,:,:]
                    # atten_get = atten_get.reshape(300,651)
                    # arr_1 = np.sum(atten_get[:150, :], axis=0)
                    # arr_2 = np.sum(atten_get[150:, :], axis=0)
                    # result = np.vstack([arr_1, arr_2])
                    # df = pd.DataFrame(result)
                    # df.to_csv('./result/attention-test.csv', index=False, header=False)
                    # df.to_csv('PBMC_pre.csv', index=False, header=False)
                    #np.savetxt(r'./result/attention-test.txt', att, delimiter=',')
                # elif dataloader == "test":
                #      v_d, v_pt, score, at = self.best_model(v_d1,v_d2, v_p,mode="eval")
                #      return v_d, v_pt, score, at;
                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()

        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / num_batches
         # if dataloader == "test":
        fpr, tpr, thresholds = roc_curve(y_label, y_pred)
        prec, recall, _ = precision_recall_curve(y_label, y_pred)
        # recall_score = recall_score(y_label, y_pred)
        #ACC1 = accuracy_score(y_label, y_pred)
        #KAPPA = cohen_kappa_score(y_label, y_pred)
        precision = tpr / (tpr + fpr)
        f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
        thred_optim = thresholds[3:][np.argmax(f1[3:])]
        # thred_optim = thresholds[5:][np.argmax(f1[5:])]
        y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
        cm1 = confusion_matrix(y_label, y_pred_s)
        accuracy2 = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
        sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
        specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
        if self.experiment:
            self.experiment.log_curve("test_roc curve", fpr, tpr)
            self.experiment.log_curve("test_pr curve", recall, prec)
        precision1 = precision_score(y_label, y_pred_s)
        ACC1 = accuracy_score(y_label, y_pred_s)
        KAPPA = cohen_kappa_score(y_label, y_pred_s)
        #return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy2, test_loss, thred_optim, precision1,att,index BAN
        return auroc, auprc, np.max(f1[3:]), sensitivity, specificity, accuracy2, test_loss, thred_optim, precision1, index
        # else:
        #         #     return auroc, auprc, test_loss
