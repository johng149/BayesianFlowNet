import torch
from tqdm.auto import tqdm
from src.training.discrete_loss import loss
from src.datasets.discrete_helper import beta_t, y_distribution, theta

def sample_t(batch_size, min_t=1e-6):
   return torch.clamp(torch.FloatTensor(batch_size).uniform_(0,1), min=min_t)

def train_discrete_model(model, optimizer, train_dl, epochs, accelerator, loss_tracker=[]):
    model.train()
    pbar = tqdm(range(epochs), desc="Training Discrete Model")
    for epoch in pbar:
        optimizer.zero_grad()
        ground_truth = next(train_dl).to(accelerator.device)
        batch_size, seq_len, K = ground_truth.shape
        t = sample_t(batch_size).to(accelerator.device)
        beta_1 = torch.ones_like(t) * 4
        beta = beta_t(beta_1, t).to(accelerator.device)
        y_prime = y_distribution(beta, K, ground_truth)
        model_input = theta(y_prime)
        output = model.forward(model_input, t)
        l = loss(beta_1, t, ground_truth, model_output_logits=output)
        accelerator.backward(l)
        optimizer.step()
        pbar.set_postfix({"loss": l.item()})
        loss_tracker.append(l.item())
    return loss_tracker