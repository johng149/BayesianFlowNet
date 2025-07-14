import torch
from src.datasets.discrete_synthetic.discrete_synthetic import DiscreteSyntheticDataset
from src.tokenizers.discrete_synthetic.discrete_synthetic_tokenizer import DiscreteSyntheticTokenizer
from src.nn.models.discrete_model import DiscreteModel
from src.training.training import train_discrete_model
from matplotlib import pyplot as plt
from src.inference.discrete_inference import dis_t, bayesian_inference

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = DiscreteSyntheticTokenizer()
max_seq_len = 32
train_ds = DiscreteSyntheticDataset(tokenizer, tokenized_length=max_seq_len)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

def make_infinite(dl):
    while True:
        for data in dl:
            yield data

train_dl = make_infinite(train_dl)

model = DiscreteModel(max_seq_len, tokenizer.vocab_size(), hidden_dim=64, num_heads=8)
model = model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_tracker = []

train_discrete_model(model, opt, train_dl, epochs=10_000, device=device, loss_tracker=loss_tracker)

# plot loss to file
plt.plot(loss_tracker)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('training_loss.png')
plt.show()

model_input = torch.normal(0, 1, (1, max_seq_len, tokenizer.vocab_size())).to(device)
model_input = torch.softmax(model_input, dim=-1) * 2 - 1

n = 190

for i in range(1, n+1):
  if i % 10 == 0:
    tokenizer.decode(model_input.squeeze(0))
  cur_it = torch.tensor([i], device=device)
  total_it = torch.tensor([n], device=device)
  t = dis_t(cur_it,total_it).to(device)
  dis_beta_1 = torch.ones_like(t, device=device) * 4
  model_output = model.forward(model_input, t)
  model_input = bayesian_inference(model_input, model_output, cur_it, total_it, dis_beta_1)

print("Final model result:")
print(tokenizer.decode(model_input.squeeze(0)))