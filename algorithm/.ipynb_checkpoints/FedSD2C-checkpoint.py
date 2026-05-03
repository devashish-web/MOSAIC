from algorithm.Base import BaseServer, BaseClient
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from copy import deepcopy
import random
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric
import numpy as np
class FedSD2CServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedSD2CServer, self).__init__(args, clients, model, data, logger)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=args.weight_decay)

        listy = self.data.y.tolist()
        self.num_classes= len(np.unique(listy))
    def run(self):
        for round in range(self.num_rounds):
            print("round "+str(round+1)+":")
            self.logger.write_round(round+1)
            self.sample()
            self.communicate()

            print("cid : ", end='')
            for cid in self.sampled_clients:
                print(cid, end=' ')
                for epoch in range(self.T_L):
                    self.clients[cid].train()
                self.clients[cid].distill()
                self.clients[cid].Fourier_perturb()
                self.clients[cid].init_z()
                self.clients[cid].train_decoder()
                self.clients[cid].gen_y()
                
            self.aggregate()
            self.global_evaluate()


    def aggregate(self):
        num_total_samples = sum([self.clients[cid].num_samples for cid in self.sampled_clients])
        
        for _ in range(5):
            for i, cid in enumerate(self.sampled_clients):
                for y,z,distilled_data  in zip(self.clients[cid].soft_y, self.clients[cid].syn_z,self.clients[cid].distilled_data):
                    syn_x = self.clients[cid].decoder(z)
                    data = Data(x=syn_x, edge_index=distilled_data.edge_index)
                    _,out = self.model(data)
                    y_pred= out.argmax(dim=1)
                    # 将 y 转换为 one-hot 形式
                    y_one_hot = F.one_hot(y, num_classes=self.num_classes)

                    # 如果需要将 one-hot 转为浮点型（用于计算）
                    y_one_hot = y_one_hot.float()

                    out = F.log_softmax(out,dim=1)
                    loss = F.kl_div(out,y_one_hot)
                    
                    loss.backward(retain_graph=True)
                    self.optimizer.step()

            


class FedSD2CClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedSD2CClient, self).__init__(args, model, data)
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.distilled_data = None
        self.syn_z = []
        self.soft_y = []
        self.feature_dim = self.data.x.shape[-1]
        self.encoder = EncoderGCN(self.feature_dim,[128],64).to(self.device)
        self.decoder = DecoderGCN(64,self.feature_dim).to(self.device)
        for param in self.decoder.parameters():
            param.requires_grad = True 
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def distill(self, top_k=10):
        # 获取节点特征、边索引、标签和训练掩码
        x, edge_index, y, train_mask = self.data.x, self.data.edge_index, self.data.y, self.data.train_mask
        if x.shape[0] < top_k:
            top_k = x.shape[0]
        # 确保设备一致性
        device = x.device
        edge_index = edge_index.to(device)
        train_mask = train_mask.to(device)
        y = y.to(device)

        # 将节点特征输入到模型中进行预测
        self.model.eval()
        with torch.no_grad():
            _,predictions = self.model(self.data)  # 假设模型返回节点的分类概率（logits）
        predictions = F.softmax(predictions,dim = 1)
        # 只计算训练节点的交叉熵损失
        loss = F.cross_entropy(predictions, y, reduction='none') 
        # print(loss)
        # 计算每个训练节点的重要性评分，基于损失（损失越大越重要）
        node_importance = -loss

        # 获取最重要的top_k个训练节点的索引，选择损失最大的top_k个节点
        _, top_k_indices = torch.topk(node_importance, top_k, largest=True)

        # 初始化一个小图列表
        data_list = []
        for node_idx in top_k_indices:
            mask = (edge_index[0] == node_idx) | (edge_index[1] == node_idx)
            node_neighbors = edge_index[:, mask]

            # 获取当前子图的节点集合
            node_set = torch.unique(node_neighbors)

            if node_set.numel() == 0:  # 如果是孤点
                # 创建一个包含孤点和自环的子图
                sub_x = x[node_idx].unsqueeze(0)  # 提取该节点的特征
                sub_edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)  # 自环边
                sub_data = Data(x=sub_x, edge_index=sub_edge_index)
            else:
                # 创建子图的节点特征
                sub_x = x[node_set]

                # 创建子图的边索引
                sub_edge_index, _ = torch_geometric.utils.subgraph(
                    node_set, edge_index, relabel_nodes=True
                )

                # 创建子图数据
                sub_data = Data(x=sub_x, edge_index=sub_edge_index)

            # 添加到列表中
            data_list.append(sub_data)

        # 保存结果
        self.distilled_data = data_list


    def Fourier_perturb(self):
        for data in self.distilled_data:
            X_f = torch.fft.fft(data.x,dim=0)
            freq = torch.fft.fftfreq(data.x.size(0))
            high_freq_mask = freq.abs() > 0.5  # 高频掩码
            X_f[high_freq_mask, :] *= 1.1  # 放大高频分量
            x_perturbed = torch.fft.ifft(X_f, dim=0).real  # 只取实部作为扰动后的特征
            x_perturbed = F.normalize(x_perturbed, dim=1)  # 每个节点特征归一化
            data.x = x_perturbed 

    def init_z(self):
        self.syn_z = []
        for data in self.distilled_data:
            mu,logvar = self.encoder(data.x,data.edge_index)
            z =self.reparameterize(mu,logvar)
            self.syn_z.append(z)

    def train_decoder(self):
        for _ in range(50):
            loss = 0
            for data, z in zip(self.distilled_data, self.syn_z):
                x_z = self.decoder(z)
                                
                h = self.model.layers[0](data.x,data.edge_index)
                h_z = self.model.layers[0](x_z ,data.edge_index)
                loss+= F.mse_loss(h,h_z)
            loss.backward(retain_graph=True)
            self.decoder_optimizer.step()

    def gen_y(self):
        self.soft_y=[]
        for data,z in zip(self.distilled_data, self.syn_z):
            x_z = self.decoder(z)
            new_data = Data(x_z,data.edge_index)
            _,pred = self.model(new_data)
            pred_labels = torch.argmax(pred, dim=1)  # 按列取最大值，dim=1 表示取每个样本的最大类别
            self.soft_y.append(pred_labels)



class EncoderGCN(nn.Module):
    """使用 GCN 层的图神经网络编码器"""

    def __init__(self, input_dim, hidden_dims, z_dim):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 输入层
        self.convs.append(GCNConv(input_dim, hidden_dims[0]))
        self.bns.append(nn.BatchNorm1d(hidden_dims[0]))

        # # 隐藏层
        # for i in range(1, len(hidden_dims)):
        #     self.convs.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
        #     self.bns.append(nn.BatchNorm1d(hidden_dims[i]))

        # 输出层
        self.fc_mu = nn.Linear(hidden_dims[-1], z_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], z_dim)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            if x.size(0) > 1:  # 只有在批量大于1时使用 BatchNorm
                x = bn(x)
            else:
                x = x

        # 直接输出每个节点的均值和对数方差
        mu = self.fc_mu(x)  # [num_nodes, z_dim]
        logvar = self.fc_logvar(x)  # [num_nodes, z_dim]
        return mu, logvar

class DecoderGCN(nn.Module):
    """
    图神经网络解码器，用于从潜在向量和类别信息重构节点特征。
    """

    def __init__(self, z_dim, feature_dim):
        super().__init__()
        self.z_dim = z_dim

        # MLP 用于重构节点特征
        # MLP 用于重构节点特征
        self.node_feature_decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, feature_dim)
        )

    def forward(self, z):

        reconstructed_x = self.node_feature_decoder(z)  # [num_nodes, feature_dim]

        return reconstructed_x
    
class GraphVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        reconstructed_x= self.decoder(z)
        return reconstructed_x,  mu, logvar

        


'''from algorithm.Base import BaseServer, BaseClient
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric
import numpy as np


def _payload_mb(obj):
    if obj is None:
        return 0.0
    if torch.is_tensor(obj):
        return (obj.numel() * obj.element_size()) / (1024 ** 2)
    if isinstance(obj, np.ndarray):
        return obj.nbytes / (1024 ** 2)
    if isinstance(obj, dict):
        return sum(_payload_mb(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_payload_mb(v) for v in obj)
    if isinstance(obj, (float, np.floating, int, np.integer, bool)):
        return 8 / (1024 ** 2)
    return 0.0


class FedSD2CServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedSD2CServer, self).__init__(args, clients, model, data, logger)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.01,
            weight_decay=args.weight_decay
        )

        listy = self.data.y.tolist()
        self.num_classes = len(np.unique(listy))

        # Communication accounting: client -> server only
        self.total_upload_mb = 0.0
        self.total_decoder_upload_mb = 0.0
        self.total_latent_upload_mb = 0.0
        self.total_label_upload_mb = 0.0
        self.total_graph_upload_mb = 0.0

    def _module_size_mb(self, module):
        total_bytes = 0
        for v in module.state_dict().values():
            total_bytes += v.numel() * v.element_size()
        return total_bytes / (1024 ** 2)

    def _distilled_graph_structure_mb(self, distilled_data_list):
        if distilled_data_list is None:
            return 0.0
        total_mb = 0.0
        for g in distilled_data_list:
            total_mb += _payload_mb(g.edge_index)
        return total_mb

    def run(self):
        for round in range(self.num_rounds):
            print("round " + str(round + 1) + ":")
            self.logger.write_round(round + 1)
            self.sample()
            self.communicate()

            print("cid : ", end='')
            for cid in self.sampled_clients:
                print(cid, end=' ')
                for epoch in range(self.T_L):
                    self.clients[cid].train()
                self.clients[cid].distill()
                self.clients[cid].Fourier_perturb()
                self.clients[cid].init_z()
                self.clients[cid].train_decoder()
                self.clients[cid].gen_y()

            self.aggregate()
            self.global_evaluate()

        print("=" * 60)
        print(f"Total decoder upload to server: {self.total_decoder_upload_mb:.2f} MB")
        print(f"Total latent upload to server: {self.total_latent_upload_mb:.2f} MB")
        print(f"Total label upload to server: {self.total_label_upload_mb:.2f} MB")
        print(f"Total distilled graph-structure upload to server: {self.total_graph_upload_mb:.2f} MB")
        print(f"Total client->server upload: {self.total_upload_mb:.2f} MB")

    def aggregate(self):
        round_decoder_upload_mb = 0.0
        round_latent_upload_mb = 0.0
        round_label_upload_mb = 0.0
        round_graph_upload_mb = 0.0

        for cid in self.sampled_clients:
            client = self.clients[cid]
            round_decoder_upload_mb += self._module_size_mb(client.decoder)
            round_latent_upload_mb += _payload_mb(client.syn_z)
            round_label_upload_mb += _payload_mb(client.soft_y)
            round_graph_upload_mb += self._distilled_graph_structure_mb(client.distilled_data)

        round_upload_mb = (
            round_decoder_upload_mb
            + round_latent_upload_mb
            + round_label_upload_mb
            + round_graph_upload_mb
        )

        self.total_decoder_upload_mb += round_decoder_upload_mb
        self.total_latent_upload_mb += round_latent_upload_mb
        self.total_label_upload_mb += round_label_upload_mb
        self.total_graph_upload_mb += round_graph_upload_mb
        self.total_upload_mb += round_upload_mb

        print(f"Round decoder upload to server: {round_decoder_upload_mb:.2f} MB")
        print(f"Round latent upload to server: {round_latent_upload_mb:.2f} MB")
        print(f"Round label upload to server: {round_label_upload_mb:.2f} MB")
        print(f"Round distilled graph-structure upload to server: {round_graph_upload_mb:.2f} MB")
        print(f"Round total upload to server: {round_upload_mb:.2f} MB")

        num_total_samples = sum([self.clients[cid].num_samples for cid in self.sampled_clients])

        for _ in range(5):
            for i, cid in enumerate(self.sampled_clients):
                client = self.clients[cid]
                for y, z, distilled_data in zip(client.soft_y, client.syn_z, client.distilled_data):
                    syn_x = client.decoder(z)
                    data = Data(x=syn_x, edge_index=distilled_data.edge_index)
                    _, out = self.model(data)

                    y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
                    out = F.log_softmax(out, dim=1)
                    loss = F.kl_div(out, y_one_hot)

                    loss.backward(retain_graph=True)
                    self.optimizer.step()


class FedSD2CClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedSD2CClient, self).__init__(args, model, data)
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.distilled_data = None
        self.syn_z = []
        self.soft_y = []
        self.feature_dim = self.data.x.shape[-1]
        self.encoder = EncoderGCN(self.feature_dim, [128], 64).to(self.device)
        self.decoder = DecoderGCN(64, self.feature_dim).to(self.device)
        for param in self.decoder.parameters():
            param.requires_grad = True
        self.decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def distill(self, top_k=10):
        x, edge_index, y, train_mask = self.data.x, self.data.edge_index, self.data.y, self.data.train_mask
        if x.shape[0] < top_k:
            top_k = x.shape[0]

        device = x.device
        edge_index = edge_index.to(device)
        train_mask = train_mask.to(device)
        y = y.to(device)

        self.model.eval()
        with torch.no_grad():
            _, predictions = self.model(self.data)
        predictions = F.softmax(predictions, dim=1)

        loss = F.cross_entropy(predictions, y, reduction='none')
        node_importance = -loss

        _, top_k_indices = torch.topk(node_importance, top_k, largest=True)

        data_list = []
        for node_idx in top_k_indices:
            mask = (edge_index[0] == node_idx) | (edge_index[1] == node_idx)
            node_neighbors = edge_index[:, mask]

            node_set = torch.unique(node_neighbors)

            if node_set.numel() == 0:
                sub_x = x[node_idx].unsqueeze(0)
                sub_edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
                sub_data = Data(x=sub_x, edge_index=sub_edge_index)
            else:
                sub_x = x[node_set]
                sub_edge_index, _ = torch_geometric.utils.subgraph(
                    node_set, edge_index, relabel_nodes=True
                )
                sub_data = Data(x=sub_x, edge_index=sub_edge_index)

            data_list.append(sub_data)

        self.distilled_data = data_list

    def Fourier_perturb(self):
        for data in self.distilled_data:
            X_f = torch.fft.fft(data.x, dim=0)
            freq = torch.fft.fftfreq(data.x.size(0))
            high_freq_mask = freq.abs() > 0.5
            X_f[high_freq_mask, :] *= 1.1
            x_perturbed = torch.fft.ifft(X_f, dim=0).real
            x_perturbed = F.normalize(x_perturbed, dim=1)
            data.x = x_perturbed

    def init_z(self):
        self.syn_z = []
        for data in self.distilled_data:
            mu, logvar = self.encoder(data.x, data.edge_index)
            z = self.reparameterize(mu, logvar)
            self.syn_z.append(z)

    def train_decoder(self):
        for _ in range(50):
            loss = 0
            for data, z in zip(self.distilled_data, self.syn_z):
                x_z = self.decoder(z)

                h = self.model.layers[0](data.x, data.edge_index)
                h_z = self.model.layers[0](x_z, data.edge_index)
                loss += F.mse_loss(h, h_z)
            loss.backward(retain_graph=True)
            self.decoder_optimizer.step()

    def gen_y(self):
        self.soft_y = []
        for data, z in zip(self.distilled_data, self.syn_z):
            x_z = self.decoder(z)
            new_data = Data(x_z, data.edge_index)
            _, pred = self.model(new_data)
            pred_labels = torch.argmax(pred, dim=1)
            self.soft_y.append(pred_labels)


class EncoderGCN(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(input_dim, hidden_dims[0]))
        self.bns.append(nn.BatchNorm1d(hidden_dims[0]))

        self.fc_mu = nn.Linear(hidden_dims[-1], z_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], z_dim)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            if x.size(0) > 1:
                x = bn(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class DecoderGCN(nn.Module):
    def __init__(self, z_dim, feature_dim):
        super().__init__()
        self.z_dim = z_dim
        self.node_feature_decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, feature_dim)
        )

    def forward(self, z):
        reconstructed_x = self.node_feature_decoder(z)
        return reconstructed_x


class GraphVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar'''