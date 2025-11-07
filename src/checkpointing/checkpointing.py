import json
import shutil
from pathlib import Path
from warnings import warn

import torch
from accelerate import Accelerator
from accelerate.utils import merge_fsdp_weights, save
from safetensors.torch import load_file
from torch.distributed.fsdp._optim_utils import _optim_state_dict
from torch.distributed.tensor import DTensor
from torch.nn import Module
from torch.optim import Optimizer


def save_checkpoint(
    accelerator: Accelerator, path: str | Path, seen_epochs: int
) -> None:
    """
    Save the model and optimizer state using the provided accelerator. This assumes
    that the model and optimizer states are already managed by the accelerator.

    Args:
        accelerator (Accelerator): The accelerator handling distributed training.
        path (str): The file path to save the checkpoint.
        seen_epochs (int): The number of epochs seen so far (for metadata).
    """
    accelerator.wait_for_everyone()
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(save_path))

    if accelerator.is_main_process:
        metadata = {
            "seen_epochs": seen_epochs,
            "is_fsdp": hasattr(accelerator.state, "fsdp_plugin")
            and accelerator.state.fsdp_plugin is not None,
            "num_processes": accelerator.num_processes,
        }
        metadata_path = save_path / f"metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)


def _merge_model_shards(
    model_shard_dir: str | Path, output_dir: str | Path, is_fsdp_current: bool
) -> None:
    """
    Merge model shards saved by FSDP into a single model file. Intended to be used to load
    FSDP checkpoints into non-FSDP models.

    Args:
        model_shard_dir (str | Path): Directory containing the FSDP model shards.
        output_dir (str | Path): Directory to save the merged model file.
        is_fsdp_current (bool): Whether the current environment is using FSDP.
    """
    model_shard_dir = Path(model_shard_dir)
    output_dir = Path(output_dir)

    # we cannot do anything if the model_shard_dir does not exist
    if not model_shard_dir.exists():
        raise FileNotFoundError(
            f"Model shard directory {model_shard_dir} does not exist."
        )
    # if the output_dir already exists and not empty, we do not overwrite
    if output_dir.exists() and any(output_dir.iterdir()):
        if not is_fsdp_current:
            # only one process, so if we hit this check, we can just raise error
            raise FileExistsError(
                f"Output directory {output_dir} already exists and is not empty."
            )
        else:
            # probably a false alarm as another process might have been the one to create the dir
            return

    merge_fsdp_weights(str(model_shard_dir), str(output_dir), safe_serialization=False)


def load_checkpoint(
    model: Module,
    optim: Optimizer | None,
    accelerator: Accelerator,
    path: str | Path,
    ignore_missing: bool = True,
) -> tuple[Module, Optimizer | None, int | None]:
    """
    Load the model and optimizer state dictionaries from a checkpoint using the provided accelerator.

    Args:
        model (Module): The model to load the state into.
        optim (Optimizer): The optimizer to load the state into.
        accelerator (Accelerator): The accelerator handling distributed training.
        path (str | Path): The file path to load the checkpoint from.
        ignore_missing (bool): If checkpoint metadata is missing, whether to abort loading or continue
    Returns:
        tuple: A tuple containing the loaded model, optimizer, and seen epochs (if available).

    There are a few cases to handle, namely:
    1. We saved checkpoint with non-FSDP and we are now loading it with non-FSDP.
    2. We saved checkpoint with non-FSDP and we are now loading it with FSDP.
    3. We saved checkpoint with FSDP and we are now loading it with non-FSDP.
    4. We saved checkpoint with FSDP and we are now loading it with FSDP.

    Case 1:
    Straightforward, we can just accelerator.load_state(load_directory)

    Case 2:
    We need to first load the model state with torch.load, and then have accelerator prepare the model and optimizer.
    We do not attempt to load the optimizer state as we can a strange error if we try.

    Case 3:
    We first need to merge model and optimizer shards into a new directory, and then we can just use
    accelerator.load_state on that new directory.

    Case 4:
    If the checkpoint metadata tells us that the number of accelerators is the same as what we currently have in our
    current setup, we can just use accelerator.load_state(load_directory).

    Otherwise, we need to merge the model and then it is just like Case 3.

    But in all cases, we need to prepare the model and optimizer with the accelerator prior to calling load_state.

    If optimizer is given, I am assuming loading needs to preserve optimizer state. However, that is only possible
    if the environment is the same (FSDP vs non-FSDP, number of ranks, etc). So if optimizer is given and we cannot
    preserve its state, we will raise an error.
    """
    accelerator.wait_for_everyone()

    save_path = Path(path)
    metadata_path = save_path / f"metadata.json"

    if not metadata_path.exists():
        if ignore_missing:
            warn(
                f"Checkpoint metadata file {metadata_path} does not exist. Assuming this is new training run"
            )
            model, optim = accelerator.prepare(model, optim)
            return model, optim, None
        else:
            raise FileNotFoundError(
                f"Checkpoint metadata file {metadata_path} does not exist."
            )

    # load metadata first to see if load environment is different from save environment
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    is_fsdp_saved = metadata.get("is_fsdp")
    num_processes_saved = metadata.get("num_processes")
    seen_epochs = metadata.get("seen_epochs")

    is_fsdp_current = (
        hasattr(accelerator.state, "fsdp_plugin")
        and accelerator.state.fsdp_plugin is not None
    )
    num_processes_current = accelerator.num_processes

    if not is_fsdp_saved and not is_fsdp_current:
        model, optim = accelerator.prepare(model, optim)
        accelerator.load_state(str(save_path))
    elif not is_fsdp_saved and is_fsdp_current:
        # we need to target model file inside checkpoint directory, we can do this by checking to see if
        # a file has the term "model" in it, as Accelerate saves model state in such a file
        model_file_path = None
        for f in save_path.iterdir():
            if "model" in f.name:
                model_file_path = f
                break
        if model_file_path is None:
            raise FileNotFoundError(
                "Could not find model file in checkpoint directory."
            )

        # either it was saved in ".bin" or ".safetensors" format
        if model_file_path.suffix == ".bin":
            state_dict = torch.load(model_file_path)
        elif model_file_path.suffix == ".safetensors":
            state_dict = load_file(model_file_path)
        else:
            raise ValueError(
                f"Unrecognized model file format: {model_file_path}, the suffix must be either .bin or .safetensors"
            )

        if optim is not None:
            raise RuntimeError(
                "Loading non-FSDP checkpoint into FSDP model with optimizer is not supported."
            )
        warn(
            "Loading non-FSDP checkpoint into FSDP model. Optimizer state will not be loaded."
        )

        model.load_state_dict(state_dict)
        model, optim = accelerator.prepare(model, optim)
    elif is_fsdp_saved and not is_fsdp_current:
        if optim is not None:
            raise RuntimeError(
                "Loading FSDP checkpoint into non-FSDP model with optimizer is not supported."
            )
        warn(
            "Loading FSDP checkpoint into non-FSDP model. Optimizer state will not be loaded."
        )

        _merge_model_shards(
            # pytorch_model_fsdp_0 is hard-coded because this is the subdirectory that Accelerate uses
            # when saving in FSDP
            model_shard_dir=save_path / "pytorch_model_fsdp_0",
            output_dir=save_path / "merged_model",
            is_fsdp_current=is_fsdp_current,
        )

        model.load_state_dict(
            torch.load(save_path / "merged_model" / "pytorch_model.bin")
        )

        model, optim = accelerator.prepare(model, optim)

        # and now delete the merged model directory to save space
        if accelerator.is_main_process:
            shutil.rmtree(save_path / "merged_model")
    elif is_fsdp_saved and is_fsdp_current:
        if num_processes_saved == num_processes_current:
            model, optim = accelerator.prepare(model, optim)
            accelerator.load_state(str(save_path))
        else:
            if optim is not None:
                raise RuntimeError(
                    "Loading FSDP checkpoint with different number of processes into non-FSDP model with optimizer is not supported."
                )
            warn(
                "Loading FSDP checkpoint with different number of processes. Optimizer state will not be loaded."
            )

            _merge_model_shards(
                model_shard_dir=save_path / "pytorch_model_fsdp_0",
                output_dir=save_path / "merged_model",
                is_fsdp_current=is_fsdp_current,
            )

            model.load_state_dict(
                torch.load(save_path / "merged_model" / "pytorch_model.bin")
            )

            model, optim = accelerator.prepare(model, optim)

            # and now delete the merged model directory to save space
            if accelerator.is_main_process:
                shutil.rmtree(save_path / "merged_model")
    else:
        # shouldn't be possible to reach here
        raise RuntimeError(
            f"Inconsistent checkpoint metadata. Checkpoint metadata FSDP is {is_fsdp_saved}, current FSDP is {is_fsdp_current}. "
            f"Checkpoint metadata num accelerators is {num_processes_saved}, current num accelerators is {num_processes_current}."
        )

    return model, optim, seen_epochs
