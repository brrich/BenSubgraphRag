import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_geometric.nn import MessagePassing

# Bayesian Neural Network components
class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer with softplus reparameterization for standard deviation.
    Uses mean-field Gaussian approximation for the posterior.
    """
    def __init__(self, in_features, out_features):
        super().__init__()

        # Initialize mean parameters directly
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        # Initialize rho parameters (will be transformed to sigma via softplus)
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

        # Small constant for numerical stability
        self.epsilon = 1e-8

    def reset_parameters(self):
        # Initialize mean with small values
        nn.init.normal_(self.weight_mu, mean=0, std=0.01)
        nn.init.normal_(self.bias_mu, mean=0, std=0.01)

        # Initialize rho to achieve sigma around 0.1-0.2 after softplus
        # For softplus, we need rho ≈ log(exp(sigma)-1)
        # To get sigma=0.1, we initialize with approximately -2.3
        nn.init.constant_(self.weight_rho, -2.3)
        nn.init.constant_(self.bias_rho, -2.3)

    def forward(self, x):
        # Get sigma through softplus transformation - always positive by construction
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
        """
        Calculate KL divergence between the posterior and prior distributions.
        Prior: N(0, 1)
        Posterior: N(weight_mu, weight_sigma^2) and N(bias_mu, bias_sigma^2)

        For a Gaussian posterior with mean μ and standard deviation σ,
        and a standard normal prior, the KL divergence is:
        KL(N(μ, σ^2) || N(0, 1)) = 0.5 * (μ^2 + σ^2 - log(σ^2) - 1)
        """
        # Get sigma through softplus transformation
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        # KL divergence for weights with improved numerical stability
        kl_weights = 0.5 * torch.sum(
            self.weight_mu**2 + weight_sigma**2 - torch.log(weight_sigma**2 + self.epsilon) - 1
        )

        # KL divergence for biases with improved numerical stability
        kl_biases = 0.5 * torch.sum(
            self.bias_mu**2 + bias_sigma**2 - torch.log(bias_sigma**2 + self.epsilon) - 1
        )

        return kl_weights + kl_biases



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
    Bayesian MLP with improved stability using softplus-based BayesianLinear layers
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

        # Using the improved BayesianLinear layers with softplus reparameterization
        self.pred = nn.Sequential(
            BayesianLinear(pred_in_size, emb_size*2),
            nn.ReLU(),
            BayesianLinear(emb_size*2, emb_size),
            nn.ReLU(),
            BayesianLinear(emb_size, emb_size//2),
            nn.ReLU(),
            BayesianLinear(emb_size//2, 1)
        )

        # Beta term for KL divergence annealing - adjusted for more stable training
        self.kl_beta = 0.001  # Start even lower
        self.kl_beta_step = 0.005  # Increase more gradually
        self.max_beta = 1.0

    def get_kl_divergence(self):
        """
        Calculate total KL divergence of all Bayesian layers in the model.
        This is called in elbo_loss function.
        """
        kl_div = 0.0

        # Sum KL divergence from all BayesianLinear layers in self.pred
        for layer in self.pred:
            if isinstance(layer, BayesianLinear):
                kl_div += layer.kl_divergence()

        return kl_div

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
        return self.pred(h_triple)
