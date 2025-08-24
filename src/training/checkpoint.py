import atexit
import json
import os
import shutil

import torch
from accelerate import Accelerator
from accelerate.utils import merge_fsdp_weights
from safetensors.torch import load_file
from torch import nn
from torch.optim import Optimizer


class CheckpointMetadata:
    def __init__(
        self,
        model_kwargs,
        optimizer_kwargs,
        is_fsdp: bool,
        num_accelerators: int,
        grad_clip_norm: float | None = None,
    ):
        self.model_kwargs = model_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.is_fsdp = is_fsdp
        self.num_accelerators = num_accelerators
        self.grad_clip_norm = grad_clip_norm

    def to_dict(self):
        return {
            "model_kwargs": self.model_kwargs,
            "optimizer_kwargs": self.optimizer_kwargs,
            "is_fsdp": self.is_fsdp,
            "num_accelerators": self.num_accelerators,
            "grad_clip_norm": self.grad_clip_norm,
        }


class CheckpointManager:
    def __init__(self):
        self.model: nn.Module | None = None
        self.optimizer: Optimizer | None = None
        self.accelerator: Accelerator | None = None
        self.metadata: CheckpointMetadata | None = None
        self.current_epoch = 0

    def ready(self):
        return (
            self.model is not None
            and self.optimizer is not None
            and self.accelerator is not None
            and self.metadata is not None
        )

    def prepare(self, model, optimizer, accelerator, metadata: CheckpointMetadata):
        self.model = model
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.metadata = metadata

        def cleanup_distributed():
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

        if (
            self.accelerator.num_processes  # pyright: ignore[reportOptionalMemberAccess]
            > 1
        ):
            atexit.register(cleanup_distributed)

    def save(self, save_directory, current_epoch):
        if not self.ready():
            raise RuntimeError(
                "CheckpointManager is not ready. Please prepare it with model, optimizer, accelerator, and metadata."
            )
        self.accelerator.wait_for_everyone()  # pyright: ignore[reportOptionalMemberAccess]
        self.accelerator.save_state(  # pyright: ignore[reportOptionalMemberAccess]
            save_directory
        )
        with open(os.path.join(save_directory, "metadata.json"), "w") as f:
            data = (
                self.metadata.to_dict()  # pyright: ignore[reportOptionalMemberAccess]
            )
            data["current_epoch"] = current_epoch
            json.dump(data, f)

    def merge_model_shards(self, model_shard_dir, output_dir):
        if not os.path.exists(model_shard_dir):
            raise FileNotFoundError(
                f"Model shard directory {model_shard_dir} does not exist."
            )
        if os.path.exists(output_dir) and os.listdir(output_dir):
            if not self.metadata.is_fsdp:  # pyright: ignore[reportOptionalMemberAccess]
                raise FileExistsError(
                    f"Output directory {output_dir} already exists and is not empty. Please choose a different directory."
                )
            else:
                # probably a false alarm so no need to raise an error
                # this is because if we are in fsdp and the mrege directory already exists,
                # it is probably because another process has already merged the shards
                return

        merge_fsdp_weights(model_shard_dir, output_dir, safe_serialization=False)

    def load(self, load_directory, error_if_not_exists: bool = True):
        if not self.ready():
            raise RuntimeError(
                "CheckpointManager is not ready. Please prepare it with model, optimizer, accelerator, and metadata."
            )

        if not os.path.exists(load_directory):
            if error_if_not_exists:
                raise FileNotFoundError(
                    f"Checkpoint directory {load_directory} does not exist."
                )
            else:
                self.accelerator.print(  # pyright: ignore[reportOptionalMemberAccess]
                    f"Checkpoint directory {load_directory} does not exist. Skipping load."
                )
                self.model, self.optimizer = (
                    self.accelerator.prepare(  # pyright: ignore[reportOptionalMemberAccess]
                        self.model, self.optimizer
                    )
                )
                return

        if not os.path.exists(os.path.join(load_directory, "metadata.json")):
            if error_if_not_exists:
                raise FileNotFoundError(f"Metadata file not found in {load_directory}.")
            else:
                self.accelerator.print(  # pyright: ignore[reportOptionalMemberAccess]
                    f"Metadata file not found in {load_directory}. Skipping load."
                )
                self.model, self.optimizer = (
                    self.accelerator.prepare(  # pyright: ignore[reportOptionalMemberAccess]
                        self.model, self.optimizer
                    )
                )
                return
        """
        This is where things get a little complicated. There are a few ways this can go down:

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
        """
        with open(os.path.join(load_directory, "metadata.json"), "r") as f:
            checkpoint_metadata = json.load(f)
        is_fsdp = checkpoint_metadata["is_fsdp"]
        accelerators = checkpoint_metadata["num_accelerators"]
        self.current_epoch = checkpoint_metadata["current_epoch"]
        print(f"Attempting to load checkpoint from epoch {self.current_epoch}")

        if (
            not is_fsdp
            and not self.metadata.is_fsdp  # pyright: ignore[reportOptionalMemberAccess]
        ):
            # Case 1
            self.model, self.optimizer = (
                self.accelerator.prepare(  # pyright: ignore[reportOptionalMemberAccess]
                    self.model, self.optimizer
                )
            )
            self.accelerator.load_state(  # pyright: ignore[reportOptionalMemberAccess]
                load_directory
            )
        elif (
            not is_fsdp
            and self.metadata.is_fsdp  # pyright: ignore[reportOptionalMemberAccess]
        ):
            # Case 2
            model_file_path = None
            for f in os.listdir(load_directory):
                if "model" in f:
                    model_file_path = os.path.join(load_directory, f)
                    break

            if model_file_path is None:
                raise FileNotFoundError(f"Model file not found in {load_directory}")

            if model_file_path.endswith(".bin"):
                state_dict = torch.load(model_file_path)
            elif model_file_path.endswith(".safetensors"):
                state_dict = load_file(model_file_path)
            else:
                raise ValueError(f"Unsupported model file format: {model_file_path}")
            self.model.load_state_dict(  # pyright: ignore[reportOptionalMemberAccess]
                state_dict
            )
            self.model, self.optimizer = (
                self.accelerator.prepare(  # pyright: ignore[reportOptionalMemberAccess]
                    self.model, self.optimizer
                )
            )
        elif (
            is_fsdp
            and not self.metadata.is_fsdp  # pyright: ignore[reportOptionalMemberAccess]
        ):
            # Case 3
            self.merge_model_shards(
                load_directory + "/pytorch_model_fsdp_0", load_directory + "_merged"
            )
            self.model.load_state_dict(  # pyright: ignore[reportOptionalMemberAccess]
                torch.load(
                    os.path.join(load_directory + "_merged", "pytorch_model.bin")
                )
            )
            self.model, self.optimizer = (
                self.accelerator.prepare(  # pyright: ignore[reportOptionalMemberAccess]
                    self.model, self.optimizer
                )
            )
            # now delete the merged directory
            if (
                self.accelerator.is_main_process  # pyright: ignore[reportOptionalMemberAccess]
            ):
                shutil.rmtree(load_directory + "_merged")
        elif (
            is_fsdp
            and self.metadata.is_fsdp  # pyright: ignore[reportOptionalMemberAccess]
        ):
            # Case 4
            if (
                accelerators
                == self.metadata.num_accelerators  # pyright: ignore[reportOptionalMemberAccess]
            ):
                self.model, self.optimizer = (
                    self.accelerator.prepare(  # pyright: ignore[reportOptionalMemberAccess]
                        self.model, self.optimizer
                    )
                )
                self.accelerator.load_state(  # pyright: ignore[reportOptionalMemberAccess]
                    load_directory
                )
            else:
                # merge model shards
                # note this sometimes fails / hangs due to data races, so if it fails the first time,
                # try running the script again
                self.merge_model_shards(
                    load_directory + "/pytorch_model_fsdp_0", load_directory + "_merged"
                )
                self.model.load_state_dict(  # pyright: ignore[reportOptionalMemberAccess]
                    torch.load(
                        os.path.join(load_directory + "_merged", "pytorch_model.bin")
                    )
                )
                self.model, self.optimizer = (
                    self.accelerator.prepare(  # pyright: ignore[reportOptionalMemberAccess]
                        self.model, self.optimizer
                    )
                )
                # now delete the merged directory
                if (
                    self.accelerator.is_main_process  # pyright: ignore[reportOptionalMemberAccess]
                ):
                    shutil.rmtree(load_directory + "_merged")
        else:
            # shouldn't be possible, but just in case
            raise RuntimeError(
                f"Inconsistent checkpoint metadata. Checkpoint metadata FSDP is {is_fsdp}, current FSDP is {self.metadata.is_fsdp}. "  # pyright: ignore[reportOptionalMemberAccess]
                f"Checkpoint metadata num accelerators is {accelerators}, current num accelerators is {self.metadata.num_accelerators}."  # pyright: ignore[reportOptionalMemberAccess]
            )
