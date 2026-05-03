from torch import nn
import torch.nn.functional as F
from algorithm.Base import BaseServer, BaseClient
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

class FENSServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FENSServer, self).__init__(args, clients, model, data, logger)
        listy = self.data.y.tolist()
        self.num_classes= len(np.unique(listy))
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.aggregator = SmallNN(d=4, total_clients=args.num_clients, num_classes=self.num_classes).to(self.device)
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
            for round_agg in range(500):
                print("train_aggregator_round "+str(round_agg+1)+":")
                self.communicate()
                self.train_aggregator()
                self.aggregate()
                


            self.global_evaluate()
    
    def aggregate(self):
        num_total_samples = sum([self.clients[cid].num_samples for cid in self.sampled_clients])
        for i, cid in enumerate(self.sampled_clients):
            w = self.clients[cid].num_samples / num_total_samples
            for client_param, global_param in zip(self.clients[cid].aggregator.parameters(), self.aggregator.parameters()):
                if i == 0:
                    global_param.data.copy_(w * client_param)
                else:
                    global_param.data += w * client_param

    def communicate(self):
        for cid in self.sampled_clients:
            for client_param, server_param in zip(self.clients[cid].aggregator.parameters(), self.aggregator.parameters()):
                client_param.data.copy_(server_param.data)

    def train_aggregator(self):
        for cid in self.sampled_clients:
            for epoch in range(1):
                logits_list = []
                self.clients[cid].aggregator.train()
                for cid_1 in self.sampled_clients:
                    self.clients[cid_1].model.eval()
                    _,out = self.clients[cid_1].model(self.clients[cid].data)
                    logits_list.append(out[self.clients[cid].train_aggregator_mask])
                logits_list = torch.cat(logits_list,dim = 1)
                out = F.log_softmax(self.clients[cid].aggregator(logits_list),dim = 1)
                loss = F.nll_loss(out, self.clients[cid].data.y[self.clients[cid].train_aggregator_mask])
                loss.backward()
                self.clients[cid].aggregator_optimizer.step()

    def global_evaluate(self):
        self.model.eval()
        self.aggregator.eval()
        logits_list = []
    
        with torch.no_grad():
            for cid in self.sampled_clients:
                _, out = self.clients[cid].model(self.data)
                logits_list.append(out)
    
            logits_list = torch.cat(logits_list, dim=1)
            out = F.log_softmax(self.aggregator(logits_list), dim=1)
    
            loss = F.nll_loss(out[self.data.test_mask], self.data.y[self.data.test_mask])
    
            pred = out[self.data.test_mask].max(dim=1)[1]
            true = self.data.y[self.data.test_mask]
    
            acc = pred.eq(true).sum().item() / self.data.test_mask.sum().item()
    
            macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
                true.cpu().numpy(),
                pred.cpu().numpy(),
                average="macro",
                zero_division=0
            )
    
            print("test_loss : " + format(loss.item(), ".4f"))
            self.logger.write_test_loss(loss.item())
    
            print("test_acc : " + format(acc, ".4f"))
            self.logger.write_test_acc(acc)
    
            print("macro_precision : " + format(macro_precision, ".4f"))
            print("macro_recall : " + format(macro_recall, ".4f"))
            print("macro_f1 : " + format(macro_f1, ".4f"))



class FENSClient(BaseClient):
    def __init__(self, args, model, data):
        super(FENSClient, self).__init__(args, model, data)
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.aggregator = SmallNN(d=4, total_clients=args.num_clients, num_classes=args.num_classes).to(self.device)
        self.train_model_mask, self.train_aggregator_mask = self.split_train_mask(self.data.train_mask)
        self.aggregator_optimizer = torch.optim.Adam(self.aggregator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        _,out = self.model(self.data)
        valid_indices = [i for i, mask in enumerate(self.train_model_mask) if mask]
        if len(valid_indices) > 0:
            loss = F.cross_entropy(out[self.train_model_mask], self.data.y[self.train_model_mask])
            loss.backward()
            self.optimizer.step()
        



    def split_train_mask(self, train_mask):
        valid_indices = [i for i, mask in enumerate(train_mask) if mask] 
        split_index = int(len(valid_indices) * 0.9)  

        train_model_mask = [False] * len(train_mask)  
        train_aggregator_mask = [False] * len(train_mask)  

        for i in valid_indices[:split_index]:
            train_model_mask[i] = True
        for i in valid_indices[split_index:]:
            train_aggregator_mask[i] = True

        return train_model_mask, train_aggregator_mask


class SmallNN(nn.Module):

    def __init__(self, d=4, total_clients=20, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(total_clients*num_classes, total_clients*d)
        self.fc2 = nn.Linear(total_clients*d, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x




