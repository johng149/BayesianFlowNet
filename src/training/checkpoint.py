import os

def save_checkpoint(accelerator, save_path):
    accelerator.wait_for_everyone()
    accelerator.save_state(save_path)

def load_checkpoint(accelerator, load_path):
    # first check if the checkpoint path exist
    if os.path.exists(load_path):
        accelerator.wait_for_everyone()
        accelerator.load_state(load_path)