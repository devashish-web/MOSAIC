'''from algorithm.Base import BaseServer, BaseClient
import torch
import copy


class FedProxServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedProxServer, self).__init__(args, clients, model, data, logger)

    def communicate(self):
        for cid in self.sampled_clients:
            self.clients[cid].global_model = copy.deepcopy(self.model)
            for client_param, server_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                client_param.data.copy_(server_param.data)


class FedProxClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedProxClient, self).__init__(args, model, data)
        self.global_model = None
        self.mu = args.fedprox_mu

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        embedding, out = self.model(self.data)
        loss = self.loss_fn(out[self.data.train_mask], self.data.y[self.data.train_mask])
        fedprox_reg = 0.0
        for client_param, server_param in zip(self.model.parameters(), self.global_model.parameters()):
            fedprox_reg += ((self.mu / 2) * torch.norm((client_param - server_param)) ** 2)
        loss += fedprox_reg
        loss.backward()
        self.optimizer.step()
        return loss.item()'''



from algorithm.Base import BaseServer, BaseClient
import torch
import copy


class FedProxServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedProxServer, self).__init__(args, clients, model, data, logger)
        self.total_upload_mb = 0.0

    def _model_size_mb(self, model):
        total_bytes = 0

        for p in model.parameters():
            total_bytes += p.numel() * p.element_size()

        for b in model.buffers():
            total_bytes += b.numel() * b.element_size()

        return total_bytes / (1024 ** 2)

    def communicate(self):
        for cid in self.sampled_clients:
            self.clients[cid].global_model = copy.deepcopy(self.model)
            for client_param, server_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                client_param.data.copy_(server_param.data)

    def aggregate(self):
        round_upload_mb = sum(
            self._model_size_mb(self.clients[cid].model)
            for cid in self.sampled_clients
        )

        self.total_upload_mb += round_upload_mb
        print(f"Round size sent to server: {round_upload_mb:.2f} MB")

        super().aggregate()

    def run(self):
        super().run()
        print("=" * 60)
        print(f"Total size sent to server: {self.total_upload_mb:.2f} MB")


class FedProxClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedProxClient, self).__init__(args, model, data)
        self.global_model = None
        self.mu = args.fedprox_mu

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        embedding, out = self.model(self.data)
        loss = self.loss_fn(out[self.data.train_mask], self.data.y[self.data.train_mask])

        fedprox_reg = 0.0
        for client_param, server_param in zip(self.model.parameters(), self.global_model.parameters()):
            fedprox_reg += ((self.mu / 2) * torch.norm((client_param - server_param)) ** 2)

        loss += fedprox_reg
        loss.backward()
        self.optimizer.step()
        return loss.item()