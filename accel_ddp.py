from accelerate import Accelerator

acc = Accelerator()

print(acc.device)
print(acc.state.parallelism_config)
assert acc.state.device_mesh is not None
print(acc.state.device_mesh)
print(acc.state.device_mesh["dp_replicate"])  # this should be the only key in DDP
