import os
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import KMeans

#AA_LIST = list("ACDEFGHIKLMNPQRSTVWXY")
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
DATASET = 'dualkiba-DualPGdq(work2)'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', DATASET))


def kmeans_c(featurelist, config, d_model):
#kmeans_for_c 的子过程，输入一个list，元素为某个蛋白质的平均特征表示，输出每个簇的中心点和每个簇出现的频率
    matrix = np.vstack(featurelist)
    kmeans = KMeans(n_clusters=config.TRAIN.DICT_SIZE)
    kmeans.fit(matrix)
    labels = kmeans.labels_
    cluster_centers = np.zeros((d_model, config.TRAIN.DICT_SIZE))
    prior = list()
    total = 0
    for i in range(config.TRAIN.DICT_SIZE):
        cluster_position = matrix[labels == i]
        num = cluster_position.shape[0]
        prior.append(num)
        total += num
        cluster_centers[:, i] = np.mean(cluster_position, axis=0) # 计算该簇所有样本的平均值作为该簇的最终中心
    prior = torch.tensor(prior)
    prior = prior / total # 将频率转化为概率分布（先验概率）
    cluster_dict = dict()
    cluster_dict['cluster_centers'] = cluster_centers
    cluster_dict['prior'] = prior
    cluster_dict['labels'] = labels
    return cluster_dict


def kmeans_for_c(train_config, train_df, save_path):
#生成蛋白质的全局特征
    print('start generating confounder dict & aa dict')
    torch.cuda.empty_cache()
    df = train_df
    d_model = 1280
    processed_path = os.path.join(save_path, 'processed')
    raw_path = os.path.join(save_path, 'raw')
    protein_f = open(os.path.join(processed_path, train_config.TRAIN.PR_PATH), 'rb') #直接读取preparation已经提取的特征
    pr_f = pickle.load(protein_f)
    text_f = torch.load(os.path.join(processed_path, 'prot_text_graphs.pt'))
    protein_db_csv = os.path.join(raw_path, 'protein_text.csv')
    protein_df = pd.read_csv(protein_db_csv)
    seqs = protein_df['target_sequence'].dropna().unique().tolist()
    seq_to_pr_id = {seq: i for i, seq in enumerate(seqs)}
    protein_col = 'Protein' if 'Protein' in df.columns else 'target_sequence'

    featurelist = list() #收集序列级特征表示，给kmeans用
    sequence_list = list() #收集蛋白质序列本身，与featurelist一一对应，后续用来计算每个簇的文本特征表示
    # store every aa feature
    aa_features = defaultdict(list) #收集残基级别表示，给替换用
    for protein_seq in tqdm(df[protein_col].dropna().unique(), desc='Processing'):
        protein_id = seq_to_pr_id[protein_seq]
        sr = pr_f[protein_id] #直接读取preparation提取的特征
        featurelist.append(sr.mean(0).cpu()) #序列平均给聚类用
        sequence_list.append(protein_seq) #收集蛋白质序列
        sr_2 = sr[1:-1, :].cpu() #去掉 tokenizer 添加的CLS和EOS，保留所有真实氨基酸残基
        for i in range(sr_2.size(0)):
            aa = protein_seq[i]
            vector = sr_2[i]  # Hidden state of the i-th amino acid
            aa_features[aa].append(vector) #[21, 1280]给替换用

    # Calculate the average feature for each amino acid type
    aa_avg_list = []
    for aa in AA_LIST:  # 计算21个氨基酸的平均特征向量并按字母排序，排除 B/U/O/J/Z
        vectors = torch.stack(aa_features[aa])  # Stack the list into a tensor
        avg_vector = torch.mean(vectors, dim=0)  # Compute the mean along the rows
        aa_avg_list.append(avg_vector)

    # Combine the average features of all amino acids into a single tensor
    aa_tensor = torch.stack(aa_avg_list)

    cluster_dict = kmeans_c(featurelist, train_config, d_model)
    labels = cluster_dict.pop('labels') #获取每个样本所属的簇
    text_cluster_list = [] #存放八个文本聚类中心
    for i in range(train_config.TRAIN.DICT_SIZE): #遍历8个簇，计算每个簇的文本特征
        text_vectors = [] #存放文本特征
        for j, label in enumerate(labels): #遍历每个样本，找到对应簇，计算文本特征
            if label == i: #只处理当前簇的样本，j为蛋白质在featurelist中的索引，sequence_list[j]为蛋白质序列，label为该蛋白质所属的簇，i为当前簇的索引
                text_vector = text_f.get(sequence_list[j]) #根据蛋白质序列去文本特征字典中查找
                if text_vector is not None:
                    text_vector = torch.as_tensor(text_vector).squeeze().cpu().to(dtype=torch.float32)
                    text_vectors.append(text_vector)
        if len(text_vectors) == 0:
            text_cluster_list.append(torch.zeros(1024))
        else: #加起来求平均
            text_cluster_list.append(torch.mean(torch.stack(text_vectors), dim=0))
    cluster_dict['text_cluster_centers'] = torch.stack(text_cluster_list)
    cluster_dict['aa'] = aa_tensor
    os.makedirs(processed_path, exist_ok=True)
    file = open(os.path.join(processed_path, train_config.TRAIN.C_PATH), 'wb')
    pickle.dump(cluster_dict, file)
    torch.cuda.empty_cache()
    print('finish generating confounder dict & aa dict')


class TrainConfig:
    class TRAIN:
        PR_PATH = "pr_f_1280_2000.pkl"
        C_PATH = "C_1280_2000_8.pkl"
        DICT_SIZE = 8


if __name__ == "__main__":
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'raw', 'data_train.csv'))
    kmeans_for_c(TrainConfig, train_df, DATA_PATH)
