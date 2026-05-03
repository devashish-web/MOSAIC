'''from algorithm.Base import BaseServer, BaseClient


class FedAvgServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedAvgServer, self).__init__(args, clients, model, data, logger)


class FedAvgClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedAvgClient, self).__init__(args, model, data)'''





import torch

from algorithm.Base import BaseServer, BaseClient


class FedAvgServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedAvgServer, self).__init__(args, clients, model, data, logger)
        self.total_upload_mb = 0.0

    def _model_size_mb(self, model):
        total_bytes = 0

        for p in model.parameters():
            total_bytes += p.numel() * p.element_size()

        for b in model.buffers():
            total_bytes += b.numel() * b.element_size()

        return total_bytes / (1024 ** 2)

    def aggregate(self):
        # total size uploaded by sampled clients in this round
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


class FedAvgClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedAvgClient, self).__init__(args, model, data)

