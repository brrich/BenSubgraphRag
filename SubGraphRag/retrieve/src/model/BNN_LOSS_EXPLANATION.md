# Bayesian Neural Network (BNN) Loss Explanation

## Overview of Evidence Lower Bound (ELBO) in BNNs

In our implementation, we use the negative Evidence Lower Bound (ELBO) as the loss function for training our Bayesian Neural Network. This document explains how the loss is implemented and why we use it.

## Mathematical Formulation

The ELBO provides a variational approximation to the true posterior distribution over the weights. In variational inference, we maximize:

```
ELBO = E_q(W|θ)[log p(D|W)] - KL(q(W|θ)||p(W))
```

Where:
- `q(W|θ)` is the variational posterior over weights
- `p(D|W)` is the likelihood of the data given weights
- `p(W)` is the prior over weights
- `KL(q||p)` is the Kullback-Leibler divergence

Since optimizers in deep learning frameworks minimize loss functions, we actually minimize the negative ELBO:

```
Loss = -ELBO = -E_q(W|θ)[log p(D|W)] + KL(q(W|θ)||p(W))
```

## Implementation Details

In our code (`train_uncertain_bnn.py`), the negative ELBO is implemented as:

```python
# Negative log likelihood (reconstruction loss)
nll_loss = F.binary_cross_entropy_with_logits(pred_logits, target_probs)

# KL divergence term - scale by batch size to avoid over-penalizing
kl_div = model.get_kl_divergence() / batch_size
kl_loss = kl_weight * kl_div

# Total ELBO loss (negative ELBO)
total_loss = nll_loss + kl_loss
```

Key implementation aspects:

1. **Data Likelihood Term**: The term `-E_q(W|θ)[log p(D|W)]` is represented by the negative log likelihood (`nll_loss`), which uses binary cross-entropy as our model performs a binary classification task.

2. **KL Divergence Term**: The `KL(q(W|θ)||p(W))` term is calculated analytically inside the model's `get_kl_divergence()` method. It's scaled by batch size to balance the influence of the prior relative to the data likelihood.

3. **KL Annealing**: We implement KL annealing by gradually increasing the `kl_weight` from a small value to a larger one throughout training. This helps prevent mode collapse in the early stages of training.

## Practical Considerations

Our implementation includes several practical considerations:

- **Batch Scaling**: The KL divergence is scaled by the batch size to maintain a proper balance with the data likelihood term.

- **KL Annealing**: We start with a small `kl_weight` (default: 0.00001) and gradually increase it to a larger value (default: 0.001) to prevent the KL term from dominating early in training.

- **Monte Carlo Sampling**: During training, we use a single Monte Carlo sample per batch, but during evaluation, we use multiple samples (configurable via `mc_samples` parameter) to better estimate the predictive distribution.

## Uncertainty Quantification

One of the main advantages of using a BNN is uncertainty quantification. During inference, we:

1. Sample multiple weight configurations from the learned posterior
2. Compute predictions for each sample
3. Calculate statistics (mean, standard deviation) across samples
4. Use these statistics to quantify prediction uncertainty

The uncertainty metrics are tracked during evaluation in `eval_epoch()`, including:
- Mean and maximum standard deviation across predictions
- Correlation between uncertainty and correctness
- Uncertainty in top-k predictions

## Why Negative ELBO?

Using the negative ELBO as a loss function provides several benefits:

1. It balances data fit with regularization (through the KL term)
2. It allows learning a distribution over model parameters, enabling uncertainty quantification
3. It provides a principled Bayesian approach to neural network training
4. It helps prevent overfitting through the KL regularization term

The negative ELBO is central to Bayesian deep learning as it allows us to approximate the true posterior distribution over weights while keeping the computation tractable.
