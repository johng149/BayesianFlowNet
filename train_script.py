import torch
from src.tokenizers.gpt2.gpt2_tokenizer import GPT2Tokenizer
from src.datasets.shakespeare.shakespeare import ShakespeareDataset
from src.datasets.discrete_helper import collate_fn
from src.nn.models.discrete_model import DiscreteModel
from src.training.training import train_discrete_model
from matplotlib import pyplot as plt
from src.inference.discrete_inference import dis_t, bayesian_inference
from accelerate import Accelerator
from src.training.checkpoint import CheckpointMetadata, CheckpointManager
from safetensors.torch import load_file
from transformers import AutoTokenizer

accelerator = Accelerator()
print(f"Using device: {accelerator.device}")
print(f"Num processes: {accelerator.num_processes}")
print(f"Using fsdp: {hasattr(accelerator.state, 'fsdp_plugin') and accelerator.state.fsdp_plugin is not None}")
tokenizer = GPT2Tokenizer()
max_seq_len = 32
train_ds = ShakespeareDataset(tokenizer=tokenizer, max_length=max_seq_len)
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
checkpoint_manager.load("./checkpoint/shakespeare", error_if_not_exists=False)

model, opt = checkpoint_manager.model, checkpoint_manager.optimizer
train_dl = accelerator.prepare(train_dl)

loss_tracker = []

train_discrete_model(model, opt, train_dl, epochs=2, accelerator=accelerator, loss_tracker=loss_tracker)
checkpoint_manager.save("./checkpoint/shakespeare")

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
prev_debug_data = None

for i in range(1, n+1):
  if i % 10 == 0:
    tokenizer.decode(model_input.squeeze(0))
  cur_it = torch.tensor([i], device=accelerator.device)
  total_it = torch.tensor([n], device=accelerator.device)
  t = dis_t(cur_it,total_it).to(accelerator.device)
  dis_beta_1 = torch.ones_like(t, device=accelerator.device) * 4
  
  current_model_input = model_input.clone()

  model_output = model.forward(model_input, t)
  # first check if model output contains nan values
  if torch.isnan(model_output).any():
    print(f"Something went wrong during inference at iteration {i}. Model output contains NaN values.")
    # now save the model input and output for debugging
    debug_data = {
        "current_iteration": {
            "model_input": current_model_input,
            "model_output": model_output,
            "t": t,
            "iteration": i
        }
    }
    if prev_debug_data:
        debug_data["previous_iteration"] = prev_debug_data
    
    torch.save(debug_data, f"debug_nan_{i}.pt")
    break # Stop inference after saving

  prev_debug_data = {
      "model_input": current_model_input,
      "model_output": model_output,
      "t": t,
      "iteration": i
  }
  model_input = bayesian_inference(model_input, model_output, cur_it, total_it, dis_beta_1)

print("Final model result:")
print(tokenizer.decode(model_input.squeeze(0)))