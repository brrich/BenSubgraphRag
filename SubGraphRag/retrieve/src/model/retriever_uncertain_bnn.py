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
    def __init__(self, in_features, out_features, prior_mean=0, prior_std=1.0):
        super().__init__()
        
        
        # Weight means and log-variances (for numerical stability)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-3, 0.1))
        
        # Bias means and log-variances
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(-3, 0.1))
        
        # Prior distribution parameters
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        
        # Initialize log prior and log posterior
        self.log_prior = 0
        self.log_posterior = 0

    def forward(self, x):
        # Convert rho to sigma using softplus for ensuring positivity
        weight_sigma = torch.log(1 + torch.exp(self.weight_rho))
        bias_sigma = torch.log(1 + torch.exp(self.bias_rho))
        
        # Sample weights and biases from the posterior distribution
        weight_epsilon = torch.randn_like(self.weight_mu)
        bias_epsilon = torch.randn_like(self.bias_mu)
        
        weight = self.weight_mu + weight_epsilon * weight_sigma
        bias = self.bias_mu + bias_epsilon * bias_sigma
        
        # Calculate KL divergence between posterior and prior
        self.log_prior = self._log_gaussian(weight, self.prior_mean, self.prior_std).sum() + \
                         self._log_gaussian(bias, self.prior_mean, self.prior_std).sum()
        
        self.log_posterior = self._log_gaussian_posterior(weight, self.weight_mu, weight_sigma).sum() + \
                             self._log_gaussian_posterior(bias, self.bias_mu, bias_sigma).sum()
        
        # Perform linear operation
        return F.linear(x, weight, bias)
    
    def _log_gaussian(self, x, mean, std):
        """Log density of a Gaussian."""
        return -0.5 * math.log(2 * math.pi) - torch.log(std) - (x - mean)**2 / (2 * std**2)
    
    def _log_gaussian_posterior(self, x, mu, sigma):
        """Log density of a Gaussian posterior."""
        return -0.5 * math.log(2 * math.pi) - torch.log(sigma) - (x - mu)**2 / (2 * sigma**2)
    
    def kl_divergence(self):
        """
        Calculate KL divergence between posterior and prior distributions.
        """
        return self.log_posterior - self.log_prior


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
    Bayesian Multi-Layer Perceptron with variational inference
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(BayesianLinear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(BayesianLinear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(nn.ReLU())
        
        # Output layer
        self.layers.append(BayesianLinear(hidden_sizes[-1], output_size))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def kl_divergence(self):
        """Calculate total KL divergence for all Bayesian layers"""
        kl_sum = 0
        for layer in self.layers:
            if hasattr(layer, 'kl_divergence'):
                kl_sum += layer.kl_divergence()
        return kl_sum


class Retriever(nn.Module):
    def __init__(
        self,
        emb_size,
        topic_pe,
        DDE_kwargs
    ):
        super().__init__()
        
        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        self.dde = DDE(**DDE_kwargs)
        
        pred_in_size = 4 * emb_size
        if topic_pe:
            pred_in_size += 2 * 2
        pred_in_size += 2 * 2 * (DDE_kwargs['num_rounds'] + DDE_kwargs['num_reverse_rounds'])

        # Replace the deterministic predictor with a Bayesian MLP
        self.pred = BayesianMLP(
            input_size=pred_in_size,
            hidden_sizes=[emb_size*2, emb_size, emb_size//2],
            output_size=1
        )
        
        # Track total KL divergence
        self.kl_divergence_sum = 0

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
        # Potentially memory-wise problematic
        h_r = relation_embs[r_id_tensor]

        h_triple = torch.cat([
            h_q.expand(len(h_r), -1),
            h_e[h_id_tensor],
            h_r,
            h_e[t_id_tensor]
        ], dim=1)
        
        # Run the Bayesian MLP
        output = self.pred(h_triple)
        
        # Store KL divergence for use in loss function
        self.kl_divergence_sum = self.pred.kl_divergence()
        
        return output
    
    def get_kl_divergence(self):
        """Return the current KL divergence value for the ELBO loss"""
        return self.kl_divergence_sum

    # Not needed anymore since we're using actual Bayesian inference
    # instead of MC dropout
    def eval(self):
        # Just call the parent method - we don't need to keep any layers in train mode
        super().eval()
        return self
