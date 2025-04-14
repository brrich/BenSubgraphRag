import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_geometric.nn import MessagePassing

# Bayesian Neural Network components
class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer with weight and bias posterior distributions.
    Uses mean-field Gaussian approximation for the posterior.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        
        # Initialize mean and rho (transformed to std) parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-3, 0.1))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(-3, 0.1))
        
    def forward(self, x):
        # Get standard deviation through softplus
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        
        # Sample epsilon from standard normal
        weight_epsilon = torch.randn_like(self.weight_mu)
        bias_epsilon = torch.randn_like(self.bias_mu)
        
        # Sample weights and bias using reparameterization trick
        weight = self.weight_mu + weight_epsilon * weight_sigma
        bias = self.bias_mu + bias_epsilon * bias_sigma
        
        # Standard linear transform
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """KL divergence between posterior and prior"""
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        
        kl_weight = 0.5 * (weight_sigma.pow(2) + self.weight_mu.pow(2) - 2 * torch.log(weight_sigma) - 1).sum()
        kl_bias = 0.5 * (bias_sigma.pow(2) + self.bias_mu.pow(2) - 2 * torch.log(bias_sigma) - 1).sum()
        
        return kl_weight + kl_bias


class PEConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, edge_index, x):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


class DDE(nn.Module):
    def __init__(
        self,
        num_rounds,
        num_reverse_rounds
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for _ in range(num_rounds):
            self.layers.append(PEConv())
        
        self.reverse_layers = nn.ModuleList()
        for _ in range(num_reverse_rounds):
            self.reverse_layers.append(PEConv())
    
    def forward(
        self,
        topic_entity_one_hot,
        edge_index,
        reverse_edge_index
    ):
        result_list = []
        
        h_pe = topic_entity_one_hot
        for layer in self.layers:
            h_pe = layer(edge_index, h_pe)
            result_list.append(h_pe)
        
        h_pe_rev = topic_entity_one_hot
        for layer in self.reverse_layers:
            h_pe_rev = layer(reverse_edge_index, h_pe_rev)
            result_list.append(h_pe_rev)
        
        return result_list


class BayesianMLP(nn.Module):
    """
    Simplified Bayesian MLP matching original architecture more closely
    """
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        
        self.fc1 = BayesianLinear(input_size, hidden_size)
        self.fc2 = BayesianLinear(hidden_size, output_size)
    
    def forward(self, x, num_samples=1):
        outputs = []
        for _ in range(num_samples):
            out = self.fc1(x)
            out = F.relu(out)
            out = self.fc2(out)
            outputs.append(out)
        
        return torch.stack(outputs).mean(0) if num_samples > 1 else outputs[0]
    
    def kl_divergence(self):
        return self.fc1.kl_divergence() + self.fc2.kl_divergence()


class Retriever(nn.Module):
    def __init__(
        self,
        emb_size,
        topic_pe,
        DDE_kwargs,
        num_samples=5  # Number of MC samples during forward pass
    ):
        super().__init__()
        
        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        self.dde = DDE(**DDE_kwargs)
        self.num_samples = num_samples
        
        pred_in_size = 4 * emb_size
        if topic_pe:
            pred_in_size += 2 * 2
        pred_in_size += 2 * 2 * (DDE_kwargs['num_rounds'] + DDE_kwargs['num_reverse_rounds'])

        # Simplified Bayesian predictor matching original architecture
        self.pred = BayesianMLP(
            input_size=pred_in_size,
            hidden_size=emb_size,  # Match original architecture
            output_size=1
        )
        
        # Beta term for KL divergence annealing
        self.kl_beta = 0.1
        self.kl_beta_step = 0.1  # Increment per epoch
        self.max_beta = 1.0

    def forward(
        self,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot
    ):
        device = entity_embs.device
        
        h_e = torch.cat(
            [
                entity_embs,
                self.non_text_entity_emb(
                    torch.LongTensor([0]).to(device)).expand(num_non_text_entities, -1)
            ]
        , dim=0)
        h_e_list = [h_e]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)

        edge_index = torch.stack([
            h_id_tensor,
            t_id_tensor
        ], dim=0)
        reverse_edge_index = torch.stack([
            t_id_tensor,
            h_id_tensor
        ], dim=0)
        dde_list = self.dde(topic_entity_one_hot, edge_index, reverse_edge_index)
        h_e_list.extend(dde_list)
        h_e = torch.cat(h_e_list, dim=1)

        h_q = q_emb
        h_r = relation_embs[r_id_tensor]

        h_triple = torch.cat([
            h_q.expand(len(h_r), -1),
            h_e[h_id_tensor],
            h_r,
            h_e[t_id_tensor]
        ], dim=1)
        
        # Multiple forward passes during training/inference
        return self.pred(h_triple, self.num_samples)
    
    def get_kl_divergence(self):
        """Return the scaled KL divergence for ELBO loss"""
        return self.kl_beta * self.pred.kl_divergence()
    
    def update_kl_beta(self):
        """Update beta term for KL annealing"""
        self.kl_beta = min(self.kl_beta + self.kl_beta_step, self.max_beta)

    # Not needed anymore since we're using actual Bayesian inference
    # instead of MC dropout
    def eval(self):
        # Just call the parent method - we don't need to keep any layers in train mode
        super().eval()
        return self
