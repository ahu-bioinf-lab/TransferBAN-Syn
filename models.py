import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from dgllife.model.gnn import GCN
from ban import BANLayer, CrossAttention
from torch.nn.utils.weight_norm import weight_norm
from attentions.AttentionGate import AttentionGate
from attentions.se_net import SENETLayer
import numpy as np
from torch_geometric.nn import GINConv, global_add_pool
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
   # print('==================pred_output=========================')
   # print(pred_output)
    #print('==================Sigmoid=========================')
   # print(n)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=3):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class DrugBAN(nn.Module):
    def __init__(self, **config):
        super(DrugBAN, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]
        mlp_in_dim1 = 300*128
        mlp_in_dim_without_att = 38528
        mlp_hidden_dim = 256
        mlp_out_dim1 = 128
        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)
        # DL cell featrues
        #self.fc1 = nn.Linear(11,64) #oneil数据集
        # self.fc1 = nn.Linear(151, 128)
        # self.fc2 = nn.Linear(128, 256)
        # self.fc3 = nn.Linear(256, 524)
        # self.fc4 = nn.Linear(524, 1024)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.bn3 = nn.BatchNorm1d(524)
        # self.bn4 = nn.BatchNorm1d(1024)
        # self.bn5 = nn.BatchNorm1d(256)
        # self.dropout = nn.Dropout(p=0.5)
        # self.dropout2 = nn.Dropout(p=0.5)
        self.ag = AttentionGate(1,1,32)
        self.SENET = SENETLayer(reduction_ratio=3)
        #修改12
        #消融实验，去掉注意力机制
        self.fc1 = nn.Linear(151, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc0 = nn.Linear(1024, 128)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)

        self.fc5 = nn.Linear(23, 32)
        self.fc6 = nn.Linear(32, 64)
        self.fc7 = nn.Linear(64, 128)


        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)


        self.bn5 = nn.BatchNorm1d(32)
        self.bn6 = nn.BatchNorm1d(64)
        self.bn7 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.5)

        # self.fc1 = nn.Linear(651, 1024)
        # self.fc2 = nn.Linear(1024, 2048)
        # self.fc3 = nn.Linear(2048, 1024)
        # self.fc4 = nn.Linear(1024, 1024)
        # self.bn1 = nn.BatchNorm1d(1024)
        # self.bn2 = nn.BatchNorm1d(2048)
        # self.bn3 = nn.BatchNorm1d(1024)
        # self.bn4 = nn.BatchNorm1d(1024)
        # self.fc3 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 256),
        # self.reduction = nn.Sequential(
        #     nn.fc(11, 64),
        #     nn.ReLU(),
        #     #F.normalize(2,2),
        #     nn.Linear(64, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     #F.normalize(2, 2),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     #F.normalize(2, 2),
        #     nn.Linear(256, 128),
        #     #F.normalize(2, 2),


        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        # self.cross_att = weight_norm(
        #     CrossAttention(in_dim1=drug_hidden_feats[-1],in_dim2=drug_hidden_feats[-1],k_dim = , v_dim, h_dim=mlp_in_dim, h_out=ban_heads),
        #     name='h_mat', dim=None)
        self.cross_attn = CrossAttention(in_dim1=drug_hidden_feats[-1],in_dim2=drug_hidden_feats[-1])
        # self.mlp_classifier = MLPDecoder(384, mlp_hidden_dim, mlp_out_dim, binary=out_binary)  修改mlp隐藏层之前
        self.mlp_classifier = MLPDecoder_2(384, mlp_hidden_dim, mlp_out_dim, binary=out_binary)
        # self.disease = MLPDecoder_get_cell(384,mlp_hidden_dim,mlp_out_dim)修改dis_mlp隐藏层之前
        self.DNN = DNN(416,1024) #Oneil的数据集
        self.disease = MLPDecoder_get_cell_2(384, mlp_hidden_dim, mlp_out_dim)
        # self.DNN = DNN(456, 1024)
        # self.DNN1 = DNN1(416)
        self.dropout = nn.Dropout(0.3)
        self.W = torch.nn.Parameter(torch.randn(128, 128))

    def forward(self, bg_d1, bg_d2,v_p,mode):
        v_d1 = self.drug_extractor(bg_d1)
        #v_d1 =self.SENET(v_d1)
        #v_d1 =v_d1.view(64,1,150,128)
        v_d2 = self.drug_extractor(bg_d2)
       # v_d2 = self.SENET(v_d2)
        # result = torch.matmul(v_d1, self.W)
        # # 与 V_D2 进行点积
        # v_d = torch.mul(result, v_d2)
        #v_d2 = v_d2.view(64, 1, 150, 128)
        #v_d = self.ag(v_d1, v_d2)
        # feat1,feat2 = self.DNN1(bg_d1,bg_d2)
        # batchnorm1 = nn.BatchNorm1d(416)
        # feat3 = torch.cat([bg_d1, bg_d2], 1)
        #v_d1,v_d2 = self.DNN(bg_d1,bg_d2)
        v_d, att = self.bcn(v_d1,v_d2)

       # v_d = torch.cat([v_d1, v_d2], 1)

        # v_d = v_d.view(64, -1, 128)
        #v_p = v_p.view(64,-1)
        # v_p = v_p.view(32, -1)
        # v_p_copy = v_p
        #
        # # 定义要添加的填充维度
        # padding = (0, 256 - 174)  # (前面填充的维度, 后面填充的维度)
        #
        # # 使用 pad 函数进行填充
        # v_p_copy = torch.nn.functional.pad(v_p_copy, padding, value=0)

        #v_d = feat3 + feat4

        v_p = v_p.to(torch.float32)

        #v_p = self.mlp_cell(v_p)
        #v_p = v_p.view(256,-1, 64)
        # v_d = torch.cat((v_d1,v_d2),1)
        # v_d3= v_d.cpu()
        # v_p1 = v_p.cpu()
        # v_d3 = v_d3.detach().numpy()
        # v_p1 = v_p1.detach().numpy()
        #v_p = v_p.view((-1,128))
        # v_p = F.normalize(v_p, 2, 2)
        # v_p = self.reduction(v_p)
        # v_p = self.mlp_cell(v_p)
        #test
        # if mode == "train":
        #     v_p = v_p.view(42, -1)
        # else:
        #     v_p = v_p.view(22, -1)

        v_p = v_p.view(32, -1)
        # v_p = v_p.view(64, -1)
        # v_p = v_p.view(25, -1)
        v_p_path = v_p[:, :151]
        v_p_sim = v_p[:, 151:]
        v_p =self.disease(v_p_path,v_p_sim)
        # v_p_path = self.bn1(F.relu(self.fc1(v_p_path)))
        # v_p_path = self.dropout(v_p_path)
        # v_p_path = self.bn2(F.relu(self.fc2(v_p_path)))
        # v_p_path = self.dropout(v_p_path)
        # v_p_path = self.bn3(F.relu(self.fc3(v_p_path)))
        # v_p_path = self.bn4(F.relu(self.fc4(v_p_path)))
        # v_p_path = self.bn1(F.relu(self.fc0(v_p_path)))
        #
        # v_p_sim = self.bn5(F.relu(self.fc5(v_p_sim)))
        # v_p_sim = self.bn6(F.relu(self.fc6(v_p_sim)))
        # v_p_sim = self.bn7(F.relu(self.fc7(v_p_sim)))
        # v_p = torch.cat((v_p_path, v_p_sim), 1)
        #self.dropout
        # v_p = self.dropout(v_p)
        # v_p = self.fc1(v_p)
        # v_p = self.relu(v_p)
        #v_p = self.dp(v_p)
        # v_p = self.fc2(v_p)
        # v_p = self.fc3(v_p)
        # v_p = self.fc4(v_p)
        # v_p = self.bn(v_p)
        #v_p = v_p_path

        # v_p = v_p_copy +v_p
        # v_p = self.bn1(F.relu(self.fc1(v_p)))
        # v_p = self.bn2(F.relu(self.fc2(v_p)))
        # v_p = self.bn3(F.relu(self.fc3(v_p)))
        # v_p = self.bn4(F.relu(self.fc4(v_p)))
       # v_p = v_p.view(32, -1, 128)
        #v_p = v_p.view(64, -1, 128)
        # v_p = v_p.view(64, -1)
        v_p = v_p.view(32, -1)
        # v_p = v_p.view(25, -1)
        # if mode == "train":
        #     v_d = v_d.view(42, -1)
        #
        # else:
        #     v_d = v_d.view(22, -1)
        v_d = v_d.view(32, -1)
      #  v_p = self.dropout(v_p)
        # v_p = self.fc5(v_p)
        # v_p = F.normalize(v_p, 2, 2)
        #v_p = v_p.view(2, 1)
        #v_p = nn.BatchNorm1d(128)
        #v_p = v_p.view(1, 2)
        # v_p = self.fc3(v_p)
        # v_p = self.relu(v_p)
        # v_p = self.fc3(v_p)
        # v_p = self.protein_extractor(v_p)
        # v_p = v_p.cpu()
        #v_p2 = v_p.detach().numpy()
        #v_d = torch.cat((v_d1,v_d2),1)
        #v_d = v_d.numpy()
        # v_d = F.normalize(v_d, 2, 2)
        # f1, att1 = self.bcn(v_d1, v_p)
        # f2,att2 =self.bcn(v_d1, v_d2)
        # f3, att3 = self.bcn(v_d2, v_p)
        # f = torch.cat((f1,f3),1)
        # f = self.bn5(f)
        # v_p = self.dropout2(f)
        #f  = self.cross_attn(v_d,v_p)
        #f = f.view(256,-1)
        #v_d = v_d.view(64, -1)
        #v_p =v_p.view(64,-1)
        #f = torch.cat([v_d, v_p], 1)
       # f, att = self.bcn(v_d, v_p)
       # v_d = v_d.view(-1,128)
      #  v_p = v_p.view(-1,128)
        f1 =torch.cat([v_d, v_p], 1)
        score = self.mlp_classifier(f1)
        if mode == "train":
            # return v_d, v_p, f, score
            # print('----TRAIN---attention is :', att)
            return v_d, v_p, f1, score
        elif mode == "eval":
            # print('-------attention is :',att)
            # return v_d, v_p, score, att 双线性注意力
            return v_d, v_p, score


class DNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(DNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size//2),
            nn.Linear(hidden_size//2,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
        )

    def forward(self, drug1_feat: torch.Tensor, drug2_feat: torch.Tensor):
        # feat = torch.cat([drug1_feat, drug2_feat], 1)
        drug1_feat =drug1_feat.float()
        drug2_feat = drug2_feat.float()
        out1 = self.network(drug1_feat)
        out2 = self.network(drug2_feat)
        return out1,out2

class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        #molecular_embedding = torch.sum(atom_embeddings, dim=1)
        # 或
        #node_feats = torch.mean(node_feats, dim=2)

        # x = g.ndata['feat']
       # print("batch1 size:",batch1.shape)
      #  print("node_feats size111:", node_feats.shape)
       # node_feats = global_add_pool(node_feats, batch1)
        #print("node_feats size22222:", node_feats.shape)
        batch_size = batch_graph.batch_size
        #子结构
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
       # node_feats = torch.sum(node_feats, dim=2)

        # node_feats = node_feats.view(batch_size,-1)
       # node_feats = node_feats.reshape(64,1,128)
        return node_feats


class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        # if padding:
        #     self.embedding = nn.Embedding(11, embedding_dim, padding_idx=0)
        # else:
        #     self.embedding = nn.Embedding(11, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        # v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.conv1(v)
        v = F.relu(v)
        v = self.bn1(v)
        v = self.conv2(v)
        v = F.relu(v)
        v = self.bn2(v)
        v = self.conv3(v)
        v = F.relu(v)
        v = self.bn3(v)
        v = v.view(v.size(0), v.size(2), -1)
        return v


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        # self.fc1 = nn.Linear(in_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn2 = nn.BatchNorm1d(hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, out_dim)
        # self.bn3 = nn.BatchNorm1d(out_dim)
        # self.fc4 = nn.Linear(out_dim, binary)
        # cross attention
        # self.fc1 = nn.Linear(in_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        # self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        # self.fc3 = nn.Linear(hidden_dim*2, out_dim*2)
        # self.bn3 = nn.BatchNorm1d(out_dim*2)
        # self.fc4 = nn.Linear(out_dim*2, out_dim)
        # self.bn4 = nn.BatchNorm1d(out_dim)
        # self.fc5 = nn.Linear(out_dim, binary)

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, out_dim * 2)
        self.bn3 = nn.BatchNorm1d(out_dim * 2)
        self.fc4 = nn.Linear(out_dim * 2, out_dim)
        self.bn4 = nn.BatchNorm1d(out_dim)
        self.fc5 = nn.Linear(out_dim, binary)
        self.dropout = nn.Dropout(p=0.3)
        '''消融实验 去掉注意力网络 v_p 64*128 v_d 64*300*128"
        self.fc1 = nn.Linear(in_dim, hidden_dim*4)
        self.bn1 = nn.BatchNorm1d(hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, out_dim*2)
        self.bn3 = nn.BatchNorm1d(out_dim*2)
        self.fc4 = nn.Linear(out_dim *2, out_dim)
        self.bn4 = nn.BatchNorm1d(out_dim)
        self.fc5 = nn.Linear(out_dim, binary)
        # self.fc1 = nn.Linear(in_dim, hidden_dim*4)
        # self.bn1 = nn.BatchNorm1d(hidden_dim*4)
        # self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        # self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        # self.fc3 = nn.Linear(hidden_dim*2, out_dim)
        # self.bn3 = nn.BatchNorm1d(out_dim)
        # self.fc4 = nn.Linear(out_dim, binary)'''

    def forward(self, x):
        # x = self.bn1(F.relu(self.fc1(x)))
        # x = self.bn2(F.relu(self.fc2(x)))
        # x = self.bn3(F.relu(self.fc3(x)))
        # x = self.fc4(x)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.dropout(x)
        x = self.bn4(F.relu(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

class MLPDecoder_get_cell(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLPDecoder_get_cell, self).__init__()
        self.fc1 = nn.Linear(151, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc0 = nn.Linear(1024, 128)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)

        self.fc5 = nn.Linear(23, 32)
        self.fc6 = nn.Linear(32, 64)
        self.fc7 = nn.Linear(64, 128)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)

        self.bn5 = nn.BatchNorm1d(32)
        self.bn6 = nn.BatchNorm1d(64)
        self.bn7 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, v_p_path,v_p_sim):
        v_p_path = self.bn1(F.relu(self.fc1(v_p_path)))
        v_p_path = self.dropout(v_p_path)
        v_p_path = self.bn2(F.relu(self.fc2(v_p_path)))
        v_p_path = self.dropout(v_p_path)
        v_p_path = self.bn3(F.relu(self.fc3(v_p_path)))
        v_p_path = self.dropout(v_p_path)
        v_p_path = self.bn4(F.relu(self.fc4(v_p_path)))
        v_p_path = self.dropout(v_p_path)
        v_p_path = self.bn1(F.relu(self.fc0(v_p_path)))
        v_p_path = self.dropout(v_p_path)

        v_p_sim = self.bn5(F.relu(self.fc5(v_p_sim)))
        v_p_sim = self.bn6(F.relu(self.fc6(v_p_sim)))
        v_p_sim = self.bn7(F.relu(self.fc7(v_p_sim)))
        v_p = torch.cat((v_p_path, v_p_sim), 1)
        return v_p
class MLPDecoder_get_cell_2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLPDecoder_get_cell_2, self).__init__()
        self.fc1 = nn.Linear(151, 128)
        self.fc2 = nn.Linear(128, 128)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)


        self.fc3 = nn.Linear(23, 64)
        self.fc4 = nn.Linear(64, 128)

        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, v_p_path,v_p_sim):
        v_p_path = self.bn1(F.relu(self.fc1(v_p_path)))
        v_p_path = self.dropout(v_p_path)
        v_p_path = self.bn2(F.relu(self.fc2(v_p_path)))
        # v_p_path = self.dropout(v_p_path)

        v_p_sim = self.bn3(F.relu(self.fc3(v_p_sim)))
        v_p_sim = self.dropout(v_p_sim)
        v_p_sim = self.bn4(F.relu(self.fc4(v_p_sim)))
        v_p = torch.cat((v_p_path, v_p_sim), 1)
        return v_p

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits

class MLPDecoder_2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder_2, self).__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.fc3 = nn.Linear(out_dim, binary)
        self.dropout = nn.Dropout(p=0.3)


    def forward(self, x):
        # x = self.bn1(F.relu(self.fc1(x)))
        # x = self.bn2(F.relu(self.fc2(x)))
        # x = self.bn3(F.relu(self.fc3(x)))
        # x = self.fc4(x)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]



