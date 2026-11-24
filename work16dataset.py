import os
import os.path as osp
import torch
import pickle
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset
import torch_geometric.data as DATA
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from tqdm import tqdm

'''
改动行：4、47、49、50、57、65、75、84、109、110、111
'''

fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

class PreDataset(Dataset):
    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        self.is_train = train
        super(PreDataset, self).__init__(root, transform, pre_transform, pre_filter)

        # 1. 加载原始 CSV
        csv_path = self.raw_paths[0] if train else self.raw_paths[1]
        print(f"正在加载 {'训练' if train else '测试'} 索引列表...")
        self.df = pd.read_csv(csv_path)

        print(f"正在使用 RDKit 从 SMILES 构建药物图特征...")
        all_csv_paths = [self.raw_paths[0], self.raw_paths[1]]
        df_all = pd.concat([pd.read_csv(p) for p in all_csv_paths if osp.exists(p)])
        smiles = df_all['compound_iso_smiles'].unique()
        self.drug_dict = dict()
        for smile in tqdm(smiles, total=len(smiles)):
            mol = Chem.MolFromSmiles(smile)
            g = self.mol2graph(mol)
            if g is not None:
                x, edge_index, edge_attr = g
                x = (x - x.min()) / (x.max() - x.min())
                self.drug_dict[smile] = {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr}
        print(f"药物图特征构建完成，共 {len(self.drug_dict)} 个唯一分子。")

        # 3. 蛋白质字典
        prot_pt_path = self.processed_paths[0] #读取文件
        print(f"正在加载蛋白质残基特征列表 (ESM-2) 到内存...")
        protein_f = open(prot_pt_path, 'rb')
        self.pr_f = pickle.load(protein_f)
        protein_db_csv = osp.join(self.raw_dir, 'protein_text.csv')
        protein_df = pd.read_csv(protein_db_csv)
        seqs = protein_df['target_sequence'].dropna().unique().tolist()
        self.seq_to_pr_id = {seq: i for i, seq in enumerate(seqs)}

        # 4. 蛋白质文本功能字典
        text_pt_path = self.processed_paths[1]
        print(f"正在加载蛋白质文本特征字典 (ProTrek) 到内存...")
        self.text_dict = torch.load(text_pt_path)

        self.target_len = 2000 #最大长度改为2000

    @property
    def raw_file_names(self):
        return ['data_train.csv', 'data_test.csv']

    @property
    def processed_file_names(self):
        return ['pr_f_1280_2000.pkl', 'prot_text_graphs.pt']

    def len(self):
        return len(self.df)

    def get(self, idx):
        # 1. 获取当前行的键值
        row = self.df.iloc[idx]
        smi = row['compound_iso_smiles']
        sequence = row['target_sequence']
        label = row['affinity']

        # 2. 获取药物图结构（RDKit 22维特征）
        drug_data = self.drug_dict.get(smi)
        if drug_data is None:
            drug_data = list(self.drug_dict.values())[0]

        # 3. 获取蛋白质 ESM-2 残基特征 [L, 1280] 和TAPB保存格式一致
        pr_id = self.seq_to_pr_id.get(sequence)
        prot_feat = self.pr_f[pr_id] if pr_id is not None else None
        if prot_feat is None:
            prot_feat = torch.zeros((1, 1280))

        # 4. 获取 1024 维的 ProTrek 文本特征
        text_feat = self.text_dict.get(sequence)
        if text_feat is None:
            text_feat = torch.zeros(1024)

        # 确保它是一维张量 [1024]
        if text_feat.dim() > 1:
            text_feat = text_feat.squeeze()

        # 5. 截断 ESM2 特征，Padding 在每个 batch 内按最大长度完成
        L, dim = prot_feat.shape
        if L > self.target_len:
            prot_feat = prot_feat[:self.target_len, :]
            L = self.target_len

        # 6. 组装数据并返回
        data = DATA.Data(
            x=drug_data['x'],
            edge_index=drug_data['edge_index'],
            edge_attr=drug_data['edge_attr'],
            y=torch.FloatTensor([label]),
            target=prot_feat, #原始特征矩阵
            target_text=text_feat.unsqueeze(0)
        )
        return data

    def process(self):
        pass



    def get_nodes(self, g):
        feat = []
        for n, d in g.nodes(data=True):
            h_t = []
            h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'Cl', 'S', 'Br', 'I', ]]
            h_t.append(d['a_num'])
            h_t.append(d['acceptor'])
            h_t.append(d['donor'])
            h_t.append(int(d['aromatic']))
            h_t += [int(d['hybridization'] == x) \
                    for x in (Chem.rdchem.HybridizationType.SP, \
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3)]
            h_t.append(d['num_h'])
            h_t.append(d['ExplicitValence'])
            h_t.append(d['FormalCharge'])
            h_t.append(d['ImplicitValence'])
            h_t.append(d['NumExplicitHs'])
            h_t.append(d['NumRadicalElectrons'])
            feat.append((n, h_t))
        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])
        return node_attr

    def get_edges(self, g):
        e = {}
        for n1, n2, d in g.edges(data=True):
            e_t = [int(d['b_type'] == x)
                   for x in (Chem.rdchem.BondType.SINGLE, \
                             Chem.rdchem.BondType.DOUBLE, \
                             Chem.rdchem.BondType.TRIPLE, \
                             Chem.rdchem.BondType.AROMATIC)]
            e_t.append(int(d['IsConjugated'] == False))
            e_t.append(int(d['IsConjugated'] == True))
            e[(n1, n2)] = e_t

        if len(e) == 0:
            return torch.LongTensor([[0], [0]]), torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
        return edge_index, edge_attr

    def mol2graph(self, mol):
        if mol is None:
            return None
        feats = chem_feature_factory.GetFeaturesForMol(mol)
        g = nx.DiGraph()

        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            g.add_node(i,
                       a_type=atom_i.GetSymbol(),
                       a_num=atom_i.GetAtomicNum(),
                       acceptor=0,
                       donor=0,
                       aromatic=atom_i.GetIsAromatic(),
                       hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs(),
                       ExplicitValence=atom_i.GetExplicitValence(),
                       FormalCharge=atom_i.GetFormalCharge(),
                       ImplicitValence=atom_i.GetImplicitValence(),
                       NumExplicitHs=atom_i.GetNumExplicitHs(),
                       NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
                       )

        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    g.nodes[n]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    g.nodes[n]['acceptor'] = 1

        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j,
                               b_type=e_ij.GetBondType(),
                               IsConjugated=int(e_ij.GetIsConjugated()),
                               )

        node_attr = self.get_nodes(g)
        edge_index, edge_attr = self.get_edges(g)

        return node_attr, edge_index, edge_attr

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'dualdavis-DualPGdq(work2random42)(threetext)'))

    print("开始进行数据集加载测试...")
    try:
        dataset = PreDataset(root=DATA_PATH, train=True)
        print(f"成功加载训练集！样本数: {len(dataset)}")

        sample = dataset[0]
        print(f"\n第一条数据检查:")
        print(f" - 药物节点特征形状: {sample.x.shape} (应为 [N, 22])")
        print(f" - 药物边索引形状: {sample.edge_index.shape}")
        print(f" - 药物边特征形状: {sample.edge_attr.shape} (应为 [E, 6])")
        print(f" - 蛋白质序列特征形状: {sample.target.shape}")
        print(f" - 蛋白质文本特征形状: {sample.target_text.shape}")
        print(f" - 标签 y: {sample.y}")

        print("\n所有体检项目通过！")

    except Exception as e:
        print(f"\n加载失败！错误原因: {e}")
