import torch
from tqdm.auto import tqdm
from src.training.discrete_loss import loss
from src.datasets.discrete_helper import beta_t, y_distribution, theta, sample_t
from src.training.checkpoint import CheckpointManager
from sqlite3 import OperationalError
from gradio_client.exceptions import AppError


def train_discrete_model(
    model,
    optimizer,
    train_dl,
    starting_epoch,
    epochs,
    accelerator,
    checkpoint_manager: CheckpointManager,
    save_dir: str,
    grad_clip_norm=None,
    save_every: int = 100,
):
    try:
        model.train()
        pbar = tqdm(
            range(starting_epoch, starting_epoch + epochs), desc="Training Discrete Model"
        )
        train_iter = iter(train_dl)
        for epoch in pbar:
            try:
                ground_truth = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                ground_truth = next(train_iter)
            x = ground_truth["x"]
            t = ground_truth["t"]
            beta_1 = ground_truth["beta_1"]
            model_input = ground_truth["theta"]  # batch_size, seq_len, K
            output = model.forward(model_input, t)
            l = loss(beta_1, t, x, model_output_logits=output)
            optimizer.zero_grad()
            accelerator.backward(l)
            if accelerator.sync_gradients and grad_clip_norm is not None:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            accelerator.log({"loss": l.item()}, step=epoch)
            pbar_description = f"Loss: {l.item():.4f}"
            if epoch % save_every == 0:
                pbar_description += " - Saving checkpoint"
                pbar.set_description(pbar_description)
                checkpoint_manager.save(save_dir, epoch)
            else:
                pbar.set_description(pbar_description)
        accelerator.end_training()

        checkpoint_manager.save(save_dir, epoch)
    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        checkpoint_manager.save(save_dir, epoch)
        accelerator.end_training()
        raise KeyboardInterrupt("Training interrupted by user.")
    except (OperationalError, AppError) as e:
        print(f"An error occurred during training: {e}")
        checkpoint_manager.save(save_dir, epoch)
        accelerator.end_training()
        raise e