import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein
from torch_geometric import data as DATA
import csv
from itertools import islice

import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
# from rdkit.Chem import MolFromSmiles
import networkx as nx
# from utils_test import *
# 产生原子的特征
from rdkit import DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from torch import nn
import torch
from torch.nn import functional as F

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):  # 将smile串转换成图
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def get_MACCS(smiles):
        m = Chem.MolFromSmiles(smiles)
        arr = np.zeros((1,), np.float32)
        fp = MACCSkeys.GenMACCSKeys(m)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

def calculate_molecule_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=128)
        return np.array(list(map(int, fingerprint)))
    else:
        return np.zeros(128)

actual_node = 0
class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df,df_drug, df_cell_path,df_cell_sim,max_drug_nodes=250):
    # def __init__(self, list_IDs, df, df_drug, df_cell, max_drug_nodes=150):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes
        self.cline_path = df_cell_path
        self.cline_sim = df_cell_sim

        # self.cline = df_cell
        self.drug_set = df_drug
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)


       # self.dic = dict_from_column

    def __len__(self):
        return len(self.list_IDs)


    def __getitem__(self, index):
        index = self.list_IDs[index]
        y = self.df.iloc[index]["label"]
        v_d1 = self.df.iloc[index]['drug-A-NAME']
        synergy1 = self.df.iloc[index]['label']
        v_d2 = self.df.iloc[index]['drug-B-NAME']
        #修改9
        cell_name = self.df.iloc[index]['Disease-type']

        pubchem_id1 = v_d1  # 以methotrexate为例
        pubchem_id2 = v_d2
        # df_drug = pd.read_csv('./datasets/bindingdb/large-data/drug-sim-embedding.csv', index_col='Drug')
        # v_d1_sim = df_drug.loc[v_d1].values# 假)
        # v_d1_sim = torch.tensor(v_d1_sim, dtype=torch.float32)
        # v_d2_sim = df_drug.loc[v_d2].values
        # v_d2_sim = torch.tensor(v_d2_sim, dtype=torch.float32)
        v_d1 = self.drug_set.loc[pubchem_id1]['smiles']
        v_d2 = self.drug_set.loc[pubchem_id2]['smiles']
        v_d1_fingerprint = calculate_molecule_fingerprint(v_d1)
        v_d2_fingerprint = calculate_molecule_fingerprint(v_d2)

        v_d1 = self.fc(smiles=v_d1, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        v_d2 = self.fc(smiles=v_d2, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        # v_d3 = 'C[C@@H]1CC[C@H]2C[C@@H](/C(=C/C=C/C=C/[C@H](C[C@H](C(=O)[C@@H]([C@@H](/C(=C/[C@H](C(=O)C[C@H](OC(=O)[C@@H]3CCCCN3C(=O)C(=O)[C@@]1(O2)O)[C@H](C)C[C@@H]4CC[C@H]([C@@H](C4)OC)OP(=O)(C)C)C)/C)O)OC)C)C)/C)OC'
        # v_d3 = self.fc(smiles=v_d3, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats1 = v_d1.ndata.pop('h')
        actual_node_feats2 = v_d2.ndata.pop('h')
        num_actual_nodes1 = actual_node_feats1.shape[0]
        num_actual_nodes2 = actual_node_feats2.shape[0]
        num_virtual_nodes1 = self.max_drug_nodes - num_actual_nodes1
        num_virtual_nodes2 = self.max_drug_nodes - num_actual_nodes2
        virtual_node_bit1 = torch.zeros([num_actual_nodes1, 1])
        virtual_node_bit2 = torch.zeros([num_actual_nodes2, 1])
        actual_node_feats1 = torch.cat((actual_node_feats1, virtual_node_bit1), 1)
        actual_node_feats2 = torch.cat((actual_node_feats2, virtual_node_bit2), 1)
        v_d1.ndata['h'] = actual_node_feats1
        v_d2.ndata['h'] = actual_node_feats2
        virtual_node_feat1 = torch.cat((torch.zeros(num_virtual_nodes1, 74), torch.ones(num_virtual_nodes1, 1)), 1)
        virtual_node_feat2 = torch.cat((torch.zeros(num_virtual_nodes2, 74), torch.ones(num_virtual_nodes2, 1)), 1)
        v_d1.add_nodes(num_virtual_nodes1, {"h": virtual_node_feat1})
        v_d2.add_nodes(num_virtual_nodes2, {"h": virtual_node_feat2})
        v_d1 = v_d1.add_self_loop()
        v_d2 = v_d2.add_self_loop()

        column_data_1 = self.cline_path[cell_name]
        cell_name_1 = self.cline_sim[cell_name]
        v_p_1 = column_data_1.values
        v_p_2 = cell_name_1.values
        v_p = np.append(v_p_1,v_p_2)

        return v_d1, v_d2,v_d1_fingerprint,v_d2_fingerprint, v_p, synergy1, index
    def _block(self, x):
                                     #in: [batch_size,num_filters,smi_max_len-3+1, 1]
        x = self.padding2(x)         #out:[batch_size,num_filters,smi_max_len-1, 1]
        px = self.max_pool(x)        #out:[batch_size,num_filters,(smi_max_len-1)/2, 1]

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches
