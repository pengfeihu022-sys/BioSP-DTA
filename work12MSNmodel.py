import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from torch import Tensor
from collections import OrderedDict
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU
import math

'''
MSN-DTA
'''

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=128, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GINConvNet, self).__init__()

        dim = 128
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # combined layers
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)  # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x.cuda(), data.edge_index.cuda(), data.batch.cuda()
        # target = data.target
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        # concat
        xc = x
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


class sequence_process(nn.Module):
    def __init__(self, vocab_size, embedding_num, dropout_rate=0.1, n_heads=8,
                 esm_dim=1280):
        super().__init__()

        # 1280维 降至 128维 (embedding_num = 128)
        self.FC_Q_S = nn.Sequential(
            nn.Linear(esm_dim, embedding_num),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_num, embedding_num)
        )

        self.FC_K_S = nn.Sequential(
            nn.Linear(esm_dim, embedding_num),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_num, embedding_num)
        )

        self.FC_V_S = nn.Sequential(
            nn.Linear(esm_dim, embedding_num),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_num, embedding_num)
        )

        # 缩放因子，用于标准的 Dot-Product Attention
        self.scale = math.sqrt(embedding_num)
        self.attention1 = LinkAttention(embedding_num, n_heads)
        self.linear = nn.Linear(embedding_num, 96)

    def forward(self, x, data, reset=False, pretrain=False):
        batch_size = len(x)
        max_len = 1200
        device = x[0].device

        # 处理截断与补零 (固定长度 1200)
        padded_x = torch.zeros(batch_size, max_len, 1280).to(device)

        for i in range(batch_size):
            seq_len = x[i].size(0)
            if seq_len > max_len:
                padded_x[i, :, :] = x[i][:max_len, :]
            else:
                padded_x[i, :seq_len, :] = x[i]

        # QKV 降维与自注意力融合
        Q = self.FC_Q_S(padded_x)
        K = self.FC_K_S(padded_x)
        V = self.FC_V_S(padded_x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)

        x_reduced = torch.matmul(attn_weights, V)
        protein_out, pro_attn = self.attention1(x_reduced)  # 输出形状: [Batch, 192] -> (你的代码里其实是128)

        # 最后通过线性层降维到 96
        x_final = self.linear(protein_out)

        return x_final


class NodeLevelBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)


class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = gnn.GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        data.x = F.relu(self.norm(self.conv(x, edge_index)))
        return data


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphConvBn(num_input_features, int(growth_rate * bn_size))
        self.conv2 = GraphConvBn(int(growth_rate * bn_size), growth_rate)

    def bn_function(self, data):
        concated_features = torch.cat(data.x, 1)
        data.x = concated_features
        data = self.conv1(data)
        return data

    def forward(self, data):
        if isinstance(data.x, Tensor):
            data.x = [data.x]
        data = self.bn_function(data)
        data = self.conv2(data)
        return data


class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.add_module('layer%d' % (i + 1), layer)

    def forward(self, data):
        features = [data.x]
        for name, layer in self.items():
            data = layer(data)
            features.append(data.x)
            data.x = features
        data.x = torch.cat(data.x, 1)
        return data


class GraphDenseNet(nn.Module):
    def __init__(self, num_input_features, out_dim, growth_rate=32, block_config=(3, 3, 3, 3), bn_sizes=[2, 3, 4, 4]):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', GraphConvBn(num_input_features, 32))]))
        num_input_features = 32

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers, num_input_features, growth_rate=growth_rate, bn_size=bn_sizes[i]
            )
            self.features.add_module('block%d' % (i + 1), block)
            num_input_features += int(num_layers * growth_rate)

            trans = GraphConvBn(num_input_features, num_input_features // 2)
            self.features.add_module("transition%d" % (i + 1), trans)
            num_input_features = num_input_features // 2

        self.classifer = nn.Linear(num_input_features, out_dim)

    def forward(self, data):
        mgdata = self.features(data)
        x = gnn.global_mean_pool(mgdata.x, mgdata.batch)
        x = self.classifer(x)
        return x


class MSN_DTA(nn.Module):
    def __init__(self, block_num, sm_feature, vocab_protein_size, embedding_size=128, filter_num=32, out_dim=1,
                 n_heads=8, blpretrain=False, contrast_dim=128):
        super().__init__()
        self.protein_encoder = sequence_process(vocab_protein_size, embedding_size)
        self.ligand_encoder = GraphDenseNet(num_input_features=22, out_dim=filter_num * 3, block_config=[8, 8, 8],
                                            bn_sizes=[2, 2, 2])
        self.n_heads = n_heads
        self.blpretrain = blpretrain

        # =====================================================================
        # BiLSTM 药物节点特征精炼（与原版 reset 模式一致）
        # =====================================================================
        lstm_dim = 64
        self.smile_input_fc = nn.Linear(22, lstm_dim)
        self.smile_lstm = nn.LSTM(lstm_dim, 96, num_layers=1, batch_first=True,
                                  bidirectional=True, dropout=0.1)
        self.out_node_sm = nn.Linear(192, 22)

        # =====================================================================
        # 文本特征投影层
        # =====================================================================
        self.text_dim = 1024
        self.text_projector = nn.Sequential(
            nn.Linear(self.text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # =====================================================================
        # Classifier 的输入维度
        # 96(蛋白) + 128(文本) + 96(药物) = 320 维
        # =====================================================================
        classifier_in_dim = filter_num * 3 * 2 + 128

        self.protein_contrast_projector = self._build_contrast_projector(filter_num * 3, contrast_dim)
        self.text_contrast_projector = self._build_contrast_projector(128, contrast_dim)

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim)
        )

    @staticmethod
    def _build_contrast_projector(in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim)
        )

    def _build_contrastive_views(self, protein_x, text_x):
        return {
            "protein_view": self.protein_contrast_projector(protein_x),
            "text_view": self.text_contrast_projector(text_x),
        }

    def forward(self, data, epochs, reset_epochs, return_contrast=False):
        target = data.target
        target_text = data.target_text

        if target_text.dim() == 3:
            target_text = target_text.squeeze(1)

        text_x = self.text_projector(target_text)

        # BiLSTM 精炼药物节点特征（与原版 reset 模式一致，每次前向都更新 data.x）
        x = data.x                                    # [N, 22] 手工特征
        x_refine = self.smile_input_fc(x)             # [N, 64]
        x_refine = x_refine.unsqueeze(0)              # [1, N, 64]
        x_refine, _ = self.smile_lstm(x_refine)       # [1, N, 192]
        x_refine = x_refine.squeeze(0)                # [N, 192]
        x_refine = self.out_node_sm(x_refine)         # [N, 22]
        data.x = x_refine

        if self.blpretrain:
            protein_x = self.protein_encoder(target, data, pretrain=True)
            ligand_x = self.ligand_encoder(data)

        elif epochs % reset_epochs == 0:
            protein_x = self.protein_encoder(target, data, reset=True)
            ligand_x = self.ligand_encoder(data)

        else:
            protein_x = self.protein_encoder(target, data)
            ligand_x = self.ligand_encoder(data)

        x = torch.cat([protein_x, text_x, ligand_x], dim=-1)
        pred = self.classifier(x)

        if return_contrast:
            return pred, self._build_contrastive_views(protein_x, text_x)

        return pred


class LinkAttention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(LinkAttention, self).__init__()
        self.query = nn.Linear(input_dim, n_heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x).transpose(1, 2)
        value = x
        a = self.softmax(query)

        out = torch.matmul(a, value)
        out = torch.sum(out, dim=1).squeeze()
        return out, a

class SMAttention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(SMAttention, self).__init__()
        self.query = nn.Linear(input_dim, n_heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x).transpose(1, 2)
        value = x
        a = self.softmax(query)

        out = torch.matmul(a, value)
        return out, a
