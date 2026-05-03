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
                    y_one_hot = F.one_hot(y, num_classes=self.num_classes)

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
        x, edge_index, y, train_mask = self.data.x, self.data.edge_index, self.data.y, self.data.train_mask
        if x.shape[0] < top_k:
            top_k = x.shape[0]
        device = x.device
        edge_index = edge_index.to(device)
        train_mask = train_mask.to(device)
        y = y.to(device)
        self.model.eval()
        with torch.no_grad():
            _,predictions = self.model(self.data) 
        predictions = F.softmax(predictions,dim = 1)
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
            X_f = torch.fft.fft(data.x,dim=0)
            freq = torch.fft.fftfreq(data.x.size(0))
            high_freq_mask = freq.abs() > 0.5  
            X_f[high_freq_mask, :] *= 1.1  
            x_perturbed = torch.fft.ifft(X_f, dim=0).real  
            x_perturbed = F.normalize(x_perturbed, dim=1)  
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
            else:
                x = x
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
        reconstructed_x= self.decoder(z)
        return reconstructed_x,  mu, logvar

    

   
