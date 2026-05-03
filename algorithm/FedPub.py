from algorithm.Base import BaseServer, BaseClient
import torch
import copy
import networkx as nx
from collections import defaultdict
from torch import Tensor
import torch.nn.functional as F
from backbone import get_model
import numpy as np


class FedPubServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedPubServer, self).__init__(args, clients, model, data, logger)
        self.args = args
        listy = self.data.y.tolist()
        out_dim = len(np.unique(listy))
        self.num_classes = out_dim
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.model = get_model("MaskedGCN", self.data.num_node_features, self.args.hidden_dim, self.num_classes, self.args.num_layers, self.args.dropout)
        self.model.to(self.device)
        for cid in range(len(self.clients)):
            client_model = get_model("MaskedGCN", self.data.num_node_features, self.args.hidden_dim, self.num_classes, self.args.num_layers, self.args.dropout)
            client_model.to(self.device)
            self.clients[cid].model = client_model
            self.clients[cid].optimizer = torch.optim.Adam(
                self.clients[cid].model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=args.weight_decay
            )

    def run(self):
        self.model = get_model("MaskedGCN", self.data.num_node_features, self.args.hidden_dim, self.num_classes, self.args.num_layers, self.args.dropout)
        self.model.to(self.device)
        for cid in range(len(self.clients)):
            client_model = get_model("MaskedGCN", self.data.num_node_features, self.args.hidden_dim, self.num_classes, self.args.num_layers, self.args.dropout)
            client_model.to(self.device)
            self.clients[cid].model = client_model
            self.clients[cid].optimizer = torch.optim.Adam(
                self.clients[cid].model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )

        for round in range(self.num_rounds):
            print("round " + str(round + 1) + ":")
            self.logger.write_round(round + 1)
            self.sample()
            self.communicate(round)

            avg_train_loss = 0
            print("sampled client id : ", end='')
            for cid in self.sampled_clients:
                print(cid, end=' ')
                for epoch in range(self.T_L):
                    self.clients[cid].round = round
                    loss = self.clients[cid].train()
                    avg_train_loss += loss * self.clients[cid].num_samples / self.num_total_samples

            print("\n")
            print("avg_train_loss = " + str(avg_train_loss))
            self.aggregate()
            self.global_evaluate()

    def communicate(self, round):
        if round >= 1:
            local_weights = []
            for cid in self.sampled_clients:
                local_weights.append([param.data.clone() for param in self.clients[cid].model.parameters()])

            sim_matrix = self.calculate_sim_matrix()
            for i, cid in enumerate(self.sampled_clients):
                aggr_local_model_weights = self.aggr_weights(local_weights, sim_matrix[i, :])
                self.clients[cid].global_model = copy.deepcopy(self.model)
                self.clients[cid].proxy = self.get_proxy_data(self.data.x.shape[1])
                for client_param, personal_param in zip(self.clients[cid].model.parameters(), aggr_local_model_weights):
                    client_param.data.copy_(personal_param.data)
        else:
            for cid in self.sampled_clients:
                self.clients[cid].global_model = copy.deepcopy(self.model)
                self.clients[cid].proxy = self.get_proxy_data(self.data.x.shape[1])
                for client_param, server_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                    client_param.data.copy_(server_param.data)

    def aggr_weights(self, local_weights, sim_row):
        if not isinstance(sim_row, torch.Tensor):
            sim_row = torch.tensor(sim_row)
        sim_row = sim_row / sim_row.sum()
        aggregated_weights = [torch.zeros_like(param) for param in local_weights[0]]
        for client_idx, client_weights in enumerate(local_weights):
            for param_idx, param in enumerate(client_weights):
                aggregated_weights[param_idx] += sim_row[client_idx] * param
        return aggregated_weights

    def aggregate(self):
        num_total_samples = sum([self.clients[cid].num_samples for cid in self.sampled_clients])
        for global_param in self.model.parameters():
            global_param.data.zero_()
        for cid in self.sampled_clients:
            w = self.clients[cid].num_samples / num_total_samples
            for client_param, global_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                global_param.data += w * client_param.data

    def calculate_sim_matrix(self):
        local_functional_embeddings = []
        for cid in self.sampled_clients:
            local_functional_embeddings.append(self.clients[cid].get_functional_embedding())

        n_connected = int(len(self.clients) * self.cl_sample_rate)
        assert n_connected == len(local_functional_embeddings)
        sim_matrix = torch.empty(n_connected, n_connected)
        for i in range(n_connected):
            for j in range(n_connected):
                sim_matrix[i, j] = 1 - F.cosine_similarity(
                    local_functional_embeddings[i].unsqueeze(0),
                    local_functional_embeddings[j].unsqueeze(0)
                )
        sim_matrix = torch.exp(10 * sim_matrix)
        row_sums = sim_matrix.sum(dim=1, keepdim=True)
        sim_matrix = sim_matrix / row_sums
        return sim_matrix

    def get_proxy_data(self, n_feat):
        num_graphs, num_nodes = 5, 100
        data = from_networkx(nx.random_partition_graph([num_nodes] * num_graphs, p_in=0.1, p_out=0, seed=self.args.seed))
        data.x = torch.normal(mean=0, std=1, size=(num_nodes * num_graphs, n_feat))
        return data


class FedPubClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedPubClient, self).__init__(args, model, data)
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.global_model = None
        self.proxy = None

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        _, out = self.model(self.data)
        loss = F.cross_entropy(out[self.data.train_mask], self.data.y[self.data.train_mask])

        for name, param in self.model.state_dict().items():
            if 'mask' in name:
                loss += torch.norm(param.float(), 1) * 0.001
            elif 'conv' in name or 'clsif' in name:
                loss += torch.norm(param.float() - self.global_model.state_dict()[name].float(), 2) * 0.001

        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def get_functional_embedding(self):
        self.model.eval()
        with torch.no_grad():
            proxy_in = self.proxy.to(self.device)
            proxy_out = self.model(proxy_in, is_proxy=True)
            proxy_out = proxy_out.mean(dim=0).clone().detach()
        return proxy_out


def from_networkx(G, group_node_attrs=None, group_edge_attrs=None):
    from torch_geometric.data import Data

    G = G.to_directed() if not nx.is_directed(G) else G
    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    data = defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data[str(key)].append(value)

    for key, value in G.graph.items():
        if key == 'node_default' or key == 'edge_default':
            continue
        key = f'graph_{key}' if key in node_attrs else key
        data[str(key)] = value

    for key, value in data.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
            data[key] = torch.stack(value, dim=0)
        else:
            try:
                data[key] = torch.tensor(value)
            except (ValueError, TypeError, RuntimeError):
                pass

    data['edge_index'] = edge_index.view(2, -1)
    data = Data.from_dict(data)

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data





