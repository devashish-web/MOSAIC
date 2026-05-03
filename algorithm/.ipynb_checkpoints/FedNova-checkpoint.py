'''from algorithm.Base import BaseServer, BaseClient
import torch
import copy


class FedNovaServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedNovaServer, self).__init__(args, clients, model, data, logger)
        self.args = args

    def run(self):
        for round in range(self.num_rounds):
            print("round " + str(round + 1) + ":")
            self.logger.write_round(round + 1)

            self.sample()
            self.communicate()

            avg_train_loss = 0.0
            print("cid : ", end='')

            for cid in self.sampled_clients:
                print(cid, end=' ')
                self.clients[cid].round = round

                for epoch in range(self.T_L):
                    loss = self.clients[cid].train()
                    avg_train_loss += (
                        loss * self.clients[cid].num_samples / self.num_total_samples
                    )

                self.clients[cid].cal_d()

            print("\n")
            print("avg_train_loss = " + str(avg_train_loss))

            self.aggregate()
            self.global_evaluate()

    def communicate(self):
        # send current global model to local working model
        for cid in self.sampled_clients:
            for client_param, server_param in zip(
                self.clients[cid].model.parameters(),
                self.model.parameters()
            ):
                client_param.data.copy_(server_param.data)

        # also save round-start global model into src_model
        for cid in self.sampled_clients:
            for client_src_param, server_param in zip(
                self.clients[cid].src_model.parameters(),
                self.model.parameters()
            ):
                client_src_param.data.copy_(server_param.data)

    def aggregate(self):
        num_total_samples = sum(
            [self.clients[cid].num_samples for cid in self.sampled_clients]
        )

        for cid in self.sampled_clients:
            w = self.clients[cid].num_samples / num_total_samples
            for d_param, global_param in zip(
                self.clients[cid].d.parameters(),
                self.model.parameters()
            ):
                global_param.data -= self.args.fednova_eta * w * d_param.data


class FedNovaClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedNovaClient, self).__init__(args, model, data)

        self.d = copy.deepcopy(self.model)
        for d_param in self.d.parameters():
            d_param.data.zero_()

        self.src_model = copy.deepcopy(self.model)
        for src_model_param in self.src_model.parameters():
            src_model_param.data.zero_()

        # In your Base.py, local training per round runs for T_L epochs
        self.tau = self.args.T_L

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()

        _, out = self.model(self.data)
        loss = self.loss_fn(
            out[self.data.train_mask],
            self.data.y[self.data.train_mask]
        )
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def cal_d(self):
        for d_param, src_model_param, model_param in zip(
            self.d.parameters(),
            self.src_model.parameters(),
            self.model.parameters()
        ):
            d_param.data.copy_(
                (src_model_param.data - model_param.data) / self.tau
            )'''

from algorithm.Base import BaseServer, BaseClient
import torch
import copy


class FedNovaServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedNovaServer, self).__init__(args, clients, model, data, logger)
        self.args = args
        self.total_upload_mb = 0.0

    def _model_size_mb(self, model):
        total_bytes = 0
        for p in model.parameters():
            total_bytes += p.numel() * p.element_size()
        for b in model.buffers():
            total_bytes += b.numel() * b.element_size()
        return total_bytes / (1024 ** 2)

    def run(self):
        for round in range(self.num_rounds):
            print("round " + str(round + 1) + ":")
            self.logger.write_round(round + 1)

            self.sample()
            self.communicate()

            avg_train_loss = 0.0
            print("cid : ", end='')

            for cid in self.sampled_clients:
                print(cid, end=' ')
                self.clients[cid].round = round

                for epoch in range(self.T_L):
                    loss = self.clients[cid].train()
                    avg_train_loss += (
                        loss * self.clients[cid].num_samples / self.num_total_samples
                    )

                self.clients[cid].cal_d()

            print("\n")
            print("avg_train_loss = " + str(avg_train_loss))

            self.aggregate()
            self.global_evaluate()

        print("=" * 60)
        print(f"Total size sent to server: {self.total_upload_mb:.2f} MB")

    def communicate(self):
        # send current global model to local working model
        for cid in self.sampled_clients:
            for client_param, server_param in zip(
                self.clients[cid].model.parameters(),
                self.model.parameters()
            ):
                client_param.data.copy_(server_param.data)

        # also save round-start global model into src_model
        for cid in self.sampled_clients:
            for client_src_param, server_param in zip(
                self.clients[cid].src_model.parameters(),
                self.model.parameters()
            ):
                client_src_param.data.copy_(server_param.data)

    def aggregate(self):
        round_upload_mb = sum(
            self._model_size_mb(self.clients[cid].d)
            for cid in self.sampled_clients
        )
        self.total_upload_mb += round_upload_mb
        print(f"Round size sent to server: {round_upload_mb:.2f} MB")

        num_total_samples = sum(
            [self.clients[cid].num_samples for cid in self.sampled_clients]
        )

        for cid in self.sampled_clients:
            w = self.clients[cid].num_samples / num_total_samples
            for d_param, global_param in zip(
                self.clients[cid].d.parameters(),
                self.model.parameters()
            ):
                global_param.data -= self.args.fednova_eta * w * d_param.data


class FedNovaClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedNovaClient, self).__init__(args, model, data)

        self.d = copy.deepcopy(self.model)
        for d_param in self.d.parameters():
            d_param.data.zero_()

        self.src_model = copy.deepcopy(self.model)
        for src_model_param in self.src_model.parameters():
            src_model_param.data.zero_()

        self.tau = self.args.T_L

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()

        _, out = self.model(self.data)
        loss = self.loss_fn(
            out[self.data.train_mask],
            self.data.y[self.data.train_mask]
        )
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def cal_d(self):
        for d_param, src_model_param, model_param in zip(
            self.d.parameters(),
            self.src_model.parameters(),
            self.model.parameters()
        ):
            d_param.data.copy_(
                (src_model_param.data - model_param.data) / self.tau
            )