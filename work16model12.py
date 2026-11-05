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
esm2序列与聚类算相似度增强文本特征
'''

class sequence_process(nn.Module): #参考tapb.py
    def __init__(self, vocab_size, embedding_num, dropout_rate=0.1, n_heads=8,
                 esm_dim=1280, c=None):
        super().__init__()

        self.esm_dim = esm_dim #1280维度
        self.n_heads = n_heads #8个注意力头
        if c is not None: #扰动和聚类QKV计算前的步骤
            self.register_buffer('c', c) #将聚类中心c注册为缓冲区，以便在模型保存和加载时保留
            self.c_center = c.size(0) #获取聚类中心的数量，后面用来切分
            if (esm_dim % self.c_center)!=0:
                raise RuntimeError(F"d_model % self.c_center)!=0")
            elif self.c_center != n_heads:
                raise RuntimeError(F"confounder dict self.c_center != n_heads")
            else:
                dim = int(esm_dim / self.c_center) #切八份
            self.linear_q = nn.Linear(esm_dim, esm_dim) #Q1280经过线性层后还是1280维
            self.linear_k = nn.Linear(esm_dim, dim) #K经过线性层后160
            self.linear_v = nn.Linear(esm_dim, dim) #V经过线性层后160
        else:
            self.c = None
            self.c_center = n_heads

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


    def confounder_aligment_module(self, pr_f, ci):
    #蛋白质聚类中心交叉注意力
        device = pr_f.device
        bz = pr_f.size(0)
        Q = self.linear_q(pr_f).view(bz, -1, self.c_center, self.esm_dim // self.c_center).permute(0, 2, 1, 3)
        K = self.linear_k(ci).unsqueeze(1).permute(0, 2, 1, 3)
        V = self.linear_v(ci).unsqueeze(1).permute(0, 2, 1, 3)
        A = torch.matmul(Q, K.permute(0, 1, 3, 2))
        A = F.softmax(A / torch.sqrt(torch.tensor(K.shape[1], dtype=torch.float32, device=device)), dim=-1)
        Q = torch.matmul(A, V)
        Q = Q.permute(0, 2, 1, 3).contiguous()
        Q = Q.view(bz, -1, self.esm_dim)
        Q = Q + pr_f
        return Q
    
    def encode_protein(self, pr_f): #输入蛋白质[batch_size, protein_len, 1280]
    #蛋白质编码
        if self.c is None:
            return pr_f
        bz = pr_f.size(0) #获得当前batch
        c_i = self.c.unsqueeze(0).expand(bz, -1, -1) #聚类中心做广播[8, 1280]→[batch_size, 8, 1280]
        pr_f = self.confounder_aligment_module(pr_f, c_i) #做交叉注意力
        return pr_f


    def forward(self, x, data, reset=False, pretrain=False):
        padded_x = x
        # 在1280维蛋白质特征进入原QKV降维之前，先与聚类中心做TAPB式八头注意力
        padded_x = self.encode_protein(padded_x)

        # QKV 降维与自注意力融合
        Q = self.FC_Q_S(padded_x)
        K = self.FC_K_S(padded_x)
        V = self.FC_V_S(padded_x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)

        x_reduced = torch.matmul(attn_weights, V)
        protein_out, pro_attn = self.attention1(x_reduced)  # 输出形状: [Batch, 128]

        # 线性层降维到 96
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


class BioSP_DTA(nn.Module):
    def __init__(self, block_num, sm_feature, vocab_protein_size, embedding_size=128, filter_num=32, out_dim=1,
                 n_heads=8, c=None, text_c=None, blpretrain=False):
        super().__init__()
        if c is None:
            raise RuntimeError("work16BioSPmodel12 requires sequence cluster centers c")
        if text_c is None:
            raise RuntimeError("work16BioSPmodel12 requires text cluster centers text_c")

        self.protein_encoder = sequence_process(vocab_protein_size, embedding_size, n_heads=n_heads, c=c)
        self.ligand_encoder = GraphDenseNet(num_input_features=22, out_dim=filter_num * 3, block_config=[8, 8, 8],
                                            bn_sizes=[2, 2, 2])
        self.n_heads = n_heads
        self.blpretrain = blpretrain

        lstm_dim = 64
        self.smile_input_fc = nn.Linear(22, lstm_dim)
        self.smile_lstm = nn.LSTM(lstm_dim, 96, num_layers=1, batch_first=True,
                                  bidirectional=True, dropout=0.1)
        self.out_node_sm = nn.Linear(192, 22)

        self.text_dim = 1024
        self.register_buffer('text_c', text_c) #保存文本聚类中心
        self.text_context_gate = nn.Sequential(
            nn.Linear(self.text_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.text_dim)
        )
        nn.init.zeros_(self.text_context_gate[-1].weight)
        nn.init.constant_(self.text_context_gate[-1].bias, -2.0)

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )


        classifier_in_dim = filter_num * 3 * 2 + 128

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

    def _masked_mean_protein(self, target, target_mask=None): #将[B, L, 1280]池化层[B, 1280]
        if target_mask is not None and target_mask.dim() == 2 and target_mask.size(1) == target.size(1):
            mask = target_mask.to(device=target.device, dtype=target.dtype)
        else:
            mask = target.abs().sum(dim=-1).gt(0).to(dtype=target.dtype)

        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (target * mask.unsqueeze(-1)).sum(dim=1) / denom

    def _enhance_text_with_cluster_context(self, target, target_text, target_mask=None):
        pr_global = self._masked_mean_protein(target, target_mask)
        c = self.protein_encoder.c.to(device=target.device, dtype=target.dtype)
        text_c = self.text_c.to(device=target_text.device, dtype=target_text.dtype)

        pr_global = F.normalize(pr_global, dim=-1)
        c = F.normalize(c, dim=-1)
        cluster_weight = F.softmax(torch.matmul(pr_global, c.transpose(0, 1)), dim=-1)
        text_context = torch.matmul(cluster_weight.to(dtype=target_text.dtype), text_c)
        gate = torch.sigmoid(self.text_context_gate(torch.cat([target_text, text_context], dim=-1)))
        return target_text + gate * text_context

    def forward(self, data, epochs, reset_epochs):
        target = data.target
        target_text = data.target_text

        if target_text.dim() == 3:
            target_text = target_text.squeeze(1)

        target_for_text = getattr(data, 'target_origin', target)
        target_mask_for_text = getattr(data, 'target_origin_mask', None)
        if target_mask_for_text is None and target_for_text is target:
            target_mask_for_text = getattr(data, 'target_mask', None)
        enhanced_text = self._enhance_text_with_cluster_context(target_for_text, target_text, target_mask_for_text)
        text_x = self.text_projector(enhanced_text)

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

            x = torch.cat([protein_x, text_x, ligand_x], dim=-1)
            x = self.classifier(x)
            return x

        elif epochs % reset_epochs == 0:
            protein_x = self.protein_encoder(target, data, reset=True)
            ligand_x = self.ligand_encoder(data)

            x = torch.cat([protein_x, text_x, ligand_x], dim=-1)
            x = self.classifier(x)
            return x

        else:
            protein_x = self.protein_encoder(target, data)
            ligand_x = self.ligand_encoder(data)

            x = torch.cat([protein_x, text_x, ligand_x], dim=-1)
            x = self.classifier(x)
            return x


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
