import torch
from src.datasets.discrete_synthetic.discrete_synthetic import DiscreteSyntheticDataset
from src.datasets.discrete_helper import collate_fn
from src.tokenizers.discrete_synthetic.discrete_synthetic_tokenizer import DiscreteSyntheticTokenizer
from src.nn.models.discrete_model import DiscreteModel
from src.training.training import train_discrete_model
from matplotlib import pyplot as plt
from src.inference.discrete_inference import dis_t, bayesian_inference
from accelerate import Accelerator
from src.training.checkpoint import CheckpointMetadata, CheckpointManager
from safetensors.torch import load_file

accelerator = Accelerator()
print(f"Using device: {accelerator.device}")
print(f"Num processes: {accelerator.num_processes}")
print(f"Using fsdp: {hasattr(accelerator.state, 'fsdp_plugin') and accelerator.state.fsdp_plugin is not None}")
tokenizer = DiscreteSyntheticTokenizer()
max_seq_len = 32
train_ds = DiscreteSyntheticDataset(tokenizer, tokenized_length=max_seq_len)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)


model_kwargs = {
    "max_seq_len": max_seq_len,
    "K": tokenizer.vocab_size(),
    "hidden_dim": 64,
    "num_heads": 8
}
model = DiscreteModel(**model_kwargs)

optimizer_kwargs = {
    "lr": 1e-3,
}
opt = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

metadata = CheckpointMetadata(
  model_kwargs=model_kwargs,
  optimizer_kwargs=optimizer_kwargs,
  is_fsdp=hasattr(accelerator.state, 'fsdp_plugin') and accelerator.state.fsdp_plugin is not None,
  num_accelerators=accelerator.num_processes
)

checkpoint_manager = CheckpointManager()
checkpoint_manager.prepare(model, opt, accelerator, metadata)
checkpoint_manager.load("./checkpoint/latest_experimental3", error_if_not_exists=False)

model, opt = checkpoint_manager.model, checkpoint_manager.optimizer
train_dl = accelerator.prepare(train_dl)

loss_tracker = []

# load_checkpoint(accelerator, "./checkpoint")
train_discrete_model(model, opt, train_dl, epochs=10, accelerator=accelerator, loss_tracker=loss_tracker)
checkpoint_manager.save("./checkpoint/latest_experimental3")

# plot loss to file
plt.plot(loss_tracker)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('training_loss.png')
plt.show()

model_input = torch.normal(0, 1, (1, max_seq_len, tokenizer.vocab_size())).to(accelerator.device)
model_input = torch.softmax(model_input, dim=-1) * 2 - 1

n = 190

for i in range(1, n+1):
  if i % 10 == 0:
    tokenizer.decode(model_input.squeeze(0))
  cur_it = torch.tensor([i], device=accelerator.device)
  total_it = torch.tensor([n], device=accelerator.device)
  t = dis_t(cur_it,total_it).to(accelerator.device)
  dis_beta_1 = torch.ones_like(t, device=accelerator.device) * 4
  model_output = model.forward(model_input, t)
  model_input = bayesian_inference(model_input, model_output, cur_it, total_it, dis_beta_1)

print("Final model result:")
print(tokenizer.decode(model_input.squeeze(0)))