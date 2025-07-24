import torch
from tqdm.auto import tqdm
from src.training.discrete_loss import loss
from src.datasets.discrete_helper import beta_t, y_distribution, theta, sample_t


def train_discrete_model(model, optimizer, train_dl, epochs, accelerator, loss_tracker=[]):
    model.train()
    pbar = tqdm(range(epochs), desc="Training Discrete Model")
    train_iter = iter(train_dl)
    for epoch in pbar:
        try:
            ground_truth = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            ground_truth = next(train_iter)
        x = ground_truth['x']
        t = ground_truth['t']
        beta_1 = ground_truth['beta_1']
        model_input = ground_truth['theta'] # batch_size, seq_len, K
        output = model.forward(model_input, t)
        l = loss(beta_1, t, x, model_output_logits=output)
        optimizer.zero_grad()
        accelerator.backward(l)
        optimizer.step()
        pbar.set_postfix({"loss": l.item()})
        loss_tracker.append(l.item())
    return loss_tracker