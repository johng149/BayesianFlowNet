from sched import scheduler

from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import AdamW as Opt
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLR
from torch.utils.data import DataLoader

from src.datasets.dataset_helper import make_collate_fn
from src.datasets.shakespeare.shakespeare import ShakespeareDataset as Ds
from src.nn.discrete_model import DiscreteModel as Model
from src.schedule.vanilla import VanillaScheduler as Scheduler
from src.tokenizers.byte.byte import ByT5Tokenizer as Tk
from src.training.train import TrainingContext as Context
from src.training.train import train


def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with="tensorboard", project_dir="./runs", kwargs_handlers=[ddp_kwargs]
    )
    checkpoint_name = "shakespeare_byt5_packed_ebt"
    checkpoint_dir = "./checkpoints"
    batch_size = 256
    seq_len = 128
    min_t = 1e-8
    num_workers = 3
    hidden_size = 768
    layers = 6
    heads = 12
    tk = Tk()
    vocab_size = tk.vocab_size()
    scheduler = Scheduler(20.4054 / vocab_size)

    train_ds = Ds(tk, seq_len, min_t=min_t)
    test_ds = Ds(tk, seq_len, min_t=min_t, train=False)

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
        use_chunkers=False,
    )

    print(
        f"Created model with {sum(p.numel() for p in model.parameters())} parameters."
    )

    opt = Opt(model.parameters(), lr=1e-4)

    lr_plateau = (
        None  # ReduceLR(opt, mode="min", factor=0.5, patience=500, cooldown=50)
    )

    context = Context(
        save_file_name=checkpoint_name,
        accelerator=accelerator,
        model=model,
        scheduler=scheduler,
        optim=opt,
        lr_scheduler=lr_plateau,
        train_loader=train_dl,
        test_loader=test_dl,
        target_epochs=15_000_000,
        test_every=5_100,
        save_every=10_000,
        test_inference_steps=100,
        save_dir=checkpoint_dir,
        grad_clip_norm=1.0,
    )

    context.load(ignore_missing=True)

    train(context)


if __name__ == "__main__":
    main()
