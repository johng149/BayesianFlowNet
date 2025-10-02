from tracemalloc import start

import torch
from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin

from src.datasets.discrete_helper import collate_fn
from src.datasets.shakespeare.shakespeare import ShakespeareDataset
from src.inference.discrete_inference import bayesian_inference, dis_t
from src.nn.models.discrete_model import DiscreteModel
from src.tokenizers.byt5.byt5_tokenizer import ByT5Tokenizer as Tokenizer
from src.training.checkpoint import CheckpointManager, CheckpointMetadata
from src.training.training import TrainingContext, train_discrete_model

# dynamo = TorchDynamoPlugin(
#     backend="inductor",  # pyright: ignore[reportArgumentType]
#     mode="default",
#     fullgraph=True,
#     dynamic=True,
# )
dynamo = None
accelerator = Accelerator(
    log_with="tensorboard", project_dir="./runs", dynamo_plugin=dynamo
)
print(f"Using device: {accelerator.device}")
print(f"Num processes: {accelerator.num_processes}")
print(
    f"Using fsdp: {hasattr(accelerator.state, 'fsdp_plugin') and accelerator.state.fsdp_plugin is not None}"
)
tokenizer = Tokenizer()
max_seq_len = 32
batch_size = 168
train_ds = ShakespeareDataset(tokenizer=tokenizer, max_length=max_seq_len)
train_dl = torch.utils.data.DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
test_ds = ShakespeareDataset(tokenizer=tokenizer, max_length=max_seq_len, train=False)
test_dl = torch.utils.data.DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)


model_kwargs = {
    "max_seq_len": max_seq_len,
    "K": tokenizer.vocab_size(),
    "hidden_dim": 64,
    "num_heads": 8,
}
model = DiscreteModel(**model_kwargs)

optimizer_kwargs = {
    "lr": 1e-3,
}
opt = torch.optim.Adam(
    model.parameters(), **optimizer_kwargs  # pyright: ignore[reportArgumentType]
)

metadata = CheckpointMetadata(
    model_kwargs=model_kwargs,
    optimizer_kwargs=optimizer_kwargs,
    is_fsdp=hasattr(accelerator.state, "fsdp_plugin")
    and accelerator.state.fsdp_plugin is not None,
    num_accelerators=accelerator.num_processes,
)

accelerator.init_trackers("shakespeare")

checkpoint_manager = CheckpointManager()
checkpoint_manager.prepare(model, opt, accelerator, metadata)
checkpoint_manager.load("./checkpoint/shakespeare", error_if_not_exists=False)
start_epoch = (
    checkpoint_manager.metadata.current_epoch if checkpoint_manager.metadata else 0
)

model, opt = checkpoint_manager.model, checkpoint_manager.optimizer
train_dl, test_dl = accelerator.prepare(train_dl, test_dl)

assert model is not None

print(f"Starting epoch: {start_epoch}")

epochs = 500

train_context = TrainingContext(
    model=model,
    optimizer=opt,
    train_dl=train_dl,
    epochs=epochs,
    accelerator=accelerator,
    checkpoint_manager=checkpoint_manager,
    save_dir="./checkpoint/shakespeare",
    starting_epoch=start_epoch,
    save_every=320,
    test_every=128,
    test_inference_steps=100,
    test_dl=test_dl,
)

train_discrete_model(train_context)
