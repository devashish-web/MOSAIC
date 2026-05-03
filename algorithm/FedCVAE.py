import torch
from torch import nn
from torch_geometric.nn import GCNConv
from algorithm.Base import BaseServer, BaseClient
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import to_dense_adj, add_self_loops, dense_to_sparse
from torch_geometric.data import Data
import matplotlib.pyplot as plt
class FedCVAEServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedCVAEServer, self).__init__(args, clients, model, data, logger)
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.feature_dim = self.data.x.shape[-1]
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
                self.train_loss = []
                print(cid, end=' ')
                print("training vae")
                for epoch in range(self.T_L):
                    loss = self.clients[cid].train()
                    self.train_loss.append(loss)


            self.aggregate()
            self.global_evaluate()

    def aggregate(self):
        num_total_samples = sum([self.clients[cid].num_samples for cid in self.sampled_clients])
        for i, cid in enumerate(self.sampled_clients):
            num_gen = int(self.clients[cid].data.x.shape[0])
            generator = self.clients[cid].decoder
            for _ in range(3):
                with torch.no_grad():
                    z_noise = torch.randn((num_gen, 64), device=self.device).float()

                    c_cnt = [0] * self.num_classes
                    for class_i in range(self.num_classes):
                        c_cnt[class_i] = int(num_gen * 1 / self.num_classes)
                    c_cnt[-1] += num_gen - sum(c_cnt)

                    print(f"pseudo label distribution: {c_cnt}")
                    y_noise = torch.zeros(num_gen).to(self.device).long()
                    ptr = 0
                    for class_i in range(self.num_classes):
                        for _ in range(c_cnt[class_i]):
                            y_noise[ptr] = class_i
                            ptr += 1 
                    
                    shuffled_indices = torch.randperm(num_gen).to(self.device)
                    y_noise = y_noise[shuffled_indices]
                    

                    y_hot = F.one_hot(y_noise, num_classes=self.num_classes).float().to(self.device)
                    node_logits = generator(z_noise,y_hot)
                    node_norm = F.normalize(node_logits, p=2, dim=1)
                    adj_logits = torch.mm(node_norm, node_norm.t())
                    pseudo_graph = construct_graph(
                        node_logits, adj_logits, k=5).to(self.device)

                for _ in range(5):
                    self.model.train()
                    self.optimizer.zero_grad()
                    self.model = self.model.to(self.device)
                    _, out = self.model(pseudo_graph)
                    loss = F.cross_entropy(out, y_noise)
                    loss.backward()
                    self.optimizer.step()

    

class FedCVAEClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedCVAEClient, self).__init__(args, model, data)
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.num_gen = self.data.x.shape[0]
        self.feature_dim = self.data.x.shape[1]

        self.num_classes = args.num_classes

        self.encoder_hidden_dims = [128]
        self.z_dim = 64

        self.y_hot = F.one_hot(self.data.y, num_classes=self.num_classes).float().to(self.device)

        self.encoder = ConditionalEncoderGCN(self.feature_dim,self.encoder_hidden_dims,self.z_dim).to(self.device)
        self.decoder = ConditionalDecoderGCN(self.z_dim,self.feature_dim, self.num_classes).to(self.device)

        self.VAE = GraphVAE(self.encoder,self.decoder).to(self.device)

        self.optimizer =  torch.optim.Adam(self.VAE.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def train(self):
        self.optimizer.zero_grad()
        
        rec_x, mu,logvar = self.VAE(self.data.x,self.data.edge_index,self.y_hot)
        
        
        loss = cal_loss(rec_x[self.data.train_mask],self.data.x[self.data.train_mask],mu[self.data.train_mask],logvar[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()



class ConditionalEncoderGCN(nn.Module):

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
            x = bn(x)
            x = torch.relu(x)

        
        mu = self.fc_mu(x)  
        logvar = self.fc_logvar(x)  
        return mu, logvar

class ConditionalDecoderGCN(nn.Module):

    def __init__(self, z_dim, feature_dim, num_classes):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.emb_layer = nn.Embedding(num_classes, num_classes)
        self.node_feature_decoder = nn.Sequential(
            nn.Linear(z_dim + num_classes, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, feature_dim)
        )

    def forward(self, z, y_hot):

        z_y = torch.cat((z, y_hot), dim=1)  
        reconstructed_x = self.node_feature_decoder(z_y)  

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

    def forward(self, x, edge_index, y_hot):
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        reconstructed_x= self.decoder(z, y_hot)
        return reconstructed_x,  mu, logvar
    


def cal_loss(rec_x, ori_x, mu, logvar):
        
    cosine_sim = F.cosine_similarity(rec_x, ori_x, dim=-1)
    
    feature_distance = 1 - cosine_sim.mean()

    kl_loss = kl_divergence(mu,logvar)

    return feature_distance + kl_loss

def kl_divergence(mu, logvar):
    
    
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean()  

    return total_kld

def construct_graph(node_logits, adj_logits, k=5):

    adjacency_matrix = torch.zeros_like(adj_logits)

    topk_values, topk_indices = torch.topk(adj_logits, k=k, dim=1)

    for i in range(node_logits.shape[0]):
        adjacency_matrix[i, topk_indices[i]] = 1
    adjacency_matrix = adjacency_matrix + adjacency_matrix.t()
    adjacency_matrix[adjacency_matrix > 1] = 1
    adjacency_matrix.fill_diagonal_(1)
    edge = adjacency_matrix.long()
    edge_index, _ = dense_to_sparse(edge)
    edge_index = add_self_loops(edge_index)[0]
    data = Data(x=node_logits, edge_index=edge_index)
    return data  


def adjust_probabilities(probX, epsilon=1e-6):

    
    prob_sum = probX.sum()
    difference = prob_sum - 1.0  

    adjusted_probX = probX.clone()

    
    if difference > epsilon:
        
        max_index = probX.argmax()
        adjusted_probX[max_index] -= difference
    
    elif difference < -epsilon:
        min_index = probX.argmin()
        adjusted_probX[min_index] -= difference 
    adjusted_probX = torch.clamp(adjusted_probX, min=0)
    adjusted_probX = adjusted_probX / (adjusted_probX.sum() + epsilon)
    
    return adjusted_probX


def draw(loss,cid):
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Loss', color='blue', marker='o')
    plt.title('Loss over Training Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.savefig(f'loss_plot_{cid}.png')






