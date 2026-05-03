from algorithm.Base import BaseServer, BaseClient
import torch
import torch.nn.functional as F

class FedRCLServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedRCLServer, self).__init__(args, clients, model, data, logger)


class FedRCLClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedRCLClient, self).__init__(args, model, data)
        self.loss_fn = F.cross_entropy

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        hidden_rep , out = self.model(self.data)
        loss_ce = self.loss_fn(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss_rcl = relaxed_contrastive_loss(hidden_rep[self.data.train_mask],self.data.y[self.data.train_mask]) + relaxed_contrastive_loss(out[self.data.train_mask],self.data.y[self.data.train_mask])
        loss = loss_ce + 1e-3 * loss_rcl
        loss.backward()
        self.optimizer.step()
        return loss.item()


def relaxed_contrastive_loss(features, labels, tau=0.05, beta=1, lambda_thresh=0.7):
    """
    Implements the Relaxed Contrastive Loss (RCL) for Federated Learning.

    Args:
        features (torch.Tensor): The feature embeddings of shape [num_nodes, feature_dim].
        labels (torch.Tensor): The labels corresponding to the features of shape [num_nodes].
        tau (float): Temperature scaling parameter.
        beta (float): Relaxation parameter for intra-class positive samples.
        lambda_thresh (float): Threshold to define intra-class positive samples.
    
    Returns:
        torch.Tensor: The computed RCL loss.
    """
    # Normalize features to ensure cosine similarity
    features = F.normalize(features, p=2, dim=1)
    
    num_nodes = features.size(0)
    device = features.device
    
    # Compute pairwise cosine similarity
    similarity_matrix = torch.matmul(features, features.T)  # [num_nodes, num_nodes]
    
    # Mask to identify positive samples (same class)
    labels = labels.view(-1, 1)  # [num_nodes, 1]
    mask_pos = (labels == labels.T).float().to(device)  # [num_nodes, num_nodes]
    mask_neg = 1.0 - mask_pos  # [num_nodes, num_nodes]
    
    # Remove diagonal (self-comparison)
    mask_pos.fill_diagonal_(0)
    
    # Compute the first term (log softmax for positives)
    numerator = torch.exp(similarity_matrix / tau) * mask_pos  # [num_nodes, num_nodes]
    denominator = torch.exp(similarity_matrix / tau) * (1 - torch.eye(num_nodes, device=device))  # [num_nodes, num_nodes]
    first_term = -torch.log(numerator.sum(dim=1) / denominator.sum(dim=1) + 1e-8).mean()
    
    # Compute the second term (relaxed intra-class positive samples)
    mask_relaxed = (similarity_matrix > lambda_thresh).float() * mask_pos  # Relaxed positive mask
    relaxed_term = beta * (
        (mask_relaxed * torch.log(numerator + 1e-8)).sum(dim=1)
    ).mean()
    
    # Final loss
    loss = first_term + relaxed_term
    return loss