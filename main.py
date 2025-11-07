from sched import scheduler

from accelerate import Accelerator
from torch.optim import AdamW as Opt
from torch.utils.data import DataLoader

from src.datasets.dataset_helper import make_collate_fn
from src.datasets.shakespeare.shakespeare import ShakespeareDataset as Ds
from src.nn.discrete_model import DiscreteModel as Model
from src.schedule.vanilla import VanillaScheduler as Scheduler
from src.tokenizers.character_level.character_level import CharacterLevelTokenizer as Tk
from src.training.train import TrainingContext as Context
from src.training.train import train


def main():
    accelerator = Accelerator(log_with="tensorboard", project_dir="./runs")
    checkpoint_name = "shakespeare_char"
    checkpoint_dir = "./checkpoints"
    batch_size = 64
    seq_len = 32
    min_t = 1e-8
    num_workers = 3
    hidden_size = 768
    layers = 7
    heads = 8
    tk = Tk()
    vocab_size = tk.vocab_size()
    scheduler = Scheduler(0.5625)

    train_ds = Ds(tk, seq_len, min_t)
    test_ds = Ds(tk, seq_len, min_t, train=False)

    collate_fn = make_collate_fn(scheduler, vocab_size)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    model = Model(
        max_seq_len=seq_len,
        K=vocab_size,
        hidden_dim=hidden_size,
        num_heads=heads,
        layers=layers,
        dropout=0.1,
    )

    opt = Opt(model.parameters(), lr=1e-4)

    context = Context(
        save_file_name=checkpoint_name,
        accelerator=accelerator,
        model=model,
        scheduler=scheduler,
        optim=opt,
        train_loader=train_dl,
        test_loader=test_dl,
        target_epochs=15_000_000,
        test_every=4_000,
        save_every=5_000,
        test_inference_steps=100,
        save_dir=checkpoint_dir,
    )

    context.load(ignore_missing=True)

    train(context)


if __name__ == "__main__":
    main()
