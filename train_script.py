import torch
from src.datasets.discrete_synthetic.discrete_synthetic import DiscreteSyntheticDataset
from src.datasets.discrete_helper import collate_fn
from src.tokenizers.discrete_synthetic.discrete_synthetic_tokenizer import DiscreteSyntheticTokenizer
from src.nn.models.discrete_model import DiscreteModel
from src.training.training import train_discrete_model
from matplotlib import pyplot as plt
from src.inference.discrete_inference import dis_t, bayesian_inference
from accelerate import Accelerator
from src.training.checkpoint import save_checkpoint, load_checkpoint

accelerator = Accelerator()
tokenizer = DiscreteSyntheticTokenizer()
max_seq_len = 32
train_ds = DiscreteSyntheticDataset(tokenizer, tokenized_length=max_seq_len)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

model = DiscreteModel(max_seq_len, tokenizer.vocab_size(), hidden_dim=64, num_heads=8)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

model, opt, train_dl = accelerator.prepare(model, opt, train_dl)

loss_tracker = []

load_checkpoint(accelerator, "./checkpoint")
train_discrete_model(model, opt, train_dl, epochs=100, accelerator=accelerator, loss_tracker=loss_tracker)
save_checkpoint(accelerator, "./checkpoint")

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