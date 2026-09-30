# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch
from transformers import TrainerCallback
from transformers.optimization import get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from . import logging


if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments

    from ..hparams import DataArguments, FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)

TT_DEVICE_TYPE = "TT"
TT_BLACKHOLE_PCI_DEVICE_ID = 0xB140
TT_REQUIRED_PACKAGES = ("torch_xla", "pjrt_plugin_tt", "torch_plugin_tt")
TT_COMPILE_OPTIONS_ENV = "LLAMAFACTORY_TT_COMPILE_OPTIONS"
TT_DEFAULT_COMPILE_OPTIONS: dict[str, Any] = {
    "fp32_dest_acc_en": True,
    "math_fidelity": "hifi4",
    "enable_const_eval": False,
}
TT_SINGLE_CHIP_MESH_DESCRIPTOR = Path(
    "tt_metal", "fabric", "mesh_graph_descriptors", "p150_mesh_graph_descriptor.textproto"
)
TT_INSTALL_HINT = (
    "pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/ "
    "(see examples/tenstorrent/README.md)"
)

_tt_environment_ready = False


def is_tt_requested() -> bool:
    r"""Check whether the user asked for the Tenstorrent backend through `PJRT_DEVICE=TT` or `USE_TT=1`."""
    if os.getenv("PJRT_DEVICE", "").upper() == TT_DEVICE_TYPE:
        return True

    return os.getenv("USE_TT", "0").lower() in ["true", "y", "1"]


@lru_cache
def _tt_packages_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in TT_REQUIRED_PACKAGES)


def is_torch_tt_available() -> bool:
    r"""Check whether training on Tenstorrent devices through PyTorch/XLA is requested and installed."""
    return is_tt_requested() and _tt_packages_available()


def _visible_tt_devices() -> list[str]:
    value = os.getenv("TT_VISIBLE_DEVICES", "").strip()
    if value:
        return [token.strip() for token in value.split(",") if token.strip()]

    device_dir = Path("/dev/tenstorrent")
    if not device_dir.is_dir():
        return []

    return sorted(node.name for node in device_dir.iterdir() if node.name.isdigit())


def _tt_pci_device_id(token: str) -> Optional[int]:
    if token.isdigit():
        path = Path(f"/sys/class/tenstorrent/tenstorrent!{token}/device/device")
    else:
        path = Path(f"/sys/bus/pci/devices/{token}/device")

    try:
        return int(path.read_text().strip(), 16)
    except (OSError, ValueError):
        return None


def _single_chip_mesh_graph_descriptor() -> Optional[str]:
    visible_devices = _visible_tt_devices()
    if len(visible_devices) != 1 or _tt_pci_device_id(visible_devices[0]) != TT_BLACKHOLE_PCI_DEVICE_ID:
        return None

    spec = importlib.util.find_spec("pjrt_plugin_tt")
    if spec is None or spec.origin is None:
        return None

    descriptor = Path(spec.origin).parent / "tt-metal" / TT_SINGLE_CHIP_MESH_DESCRIPTOR
    return str(descriptor) if descriptor.is_file() else None


def set_tt_compile_options(extra_options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    r"""Set the tt-mlir compile options, merging defaults, the JSON in `LLAMAFACTORY_TT_COMPILE_OPTIONS` and extras."""
    import torch_xla

    options = dict(TT_DEFAULT_COMPILE_OPTIONS)
    env_options = os.getenv(TT_COMPILE_OPTIONS_ENV)
    if env_options:
        options.update(json.loads(env_options))

    if extra_options:
        options.update(extra_options)

    torch_xla.set_custom_compile_options(options)
    return options


def setup_tt_environment() -> None:
    r"""Prepare the process for PyTorch/XLA on Tenstorrent devices. Must run before any XLA device is created."""
    global _tt_environment_ready
    if _tt_environment_ready:
        return

    if not _tt_packages_available():
        raise ImportError(f"Tenstorrent training requires torch_xla and pjrt-plugin-tt: {TT_INSTALL_HINT}")

    os.environ["PJRT_DEVICE"] = TT_DEVICE_TYPE
    os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if "TT_METAL_LOGS_PATH" not in os.environ:
        logs_dir = Path.home() / ".cache" / "llamafactory" / "tenstorrent"
        logs_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TT_METAL_LOGS_PATH"] = str(logs_dir)

    if "TT_MESH_GRAPH_DESC_PATH" not in os.environ:
        descriptor = _single_chip_mesh_graph_descriptor()
        if descriptor is not None:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = descriptor
            logger.info_rank0(f"One Blackhole chip is visible, using the mesh graph descriptor: {descriptor}")

    import torch_xla.runtime as xr

    xr.set_device_type(TT_DEVICE_TYPE)
    options = set_tt_compile_options()
    _tt_environment_ready = True
    logger.info_rank0(
        f"Tenstorrent backend enabled, TT_VISIBLE_DEVICES={os.getenv('TT_VISIBLE_DEVICES', '<all>')}, "
        f"compile options: {options}"
    )


def get_tt_device() -> "torch.device":
    r"""Return the XLA device, preparing the environment first so the runtime opens the chip correctly."""
    setup_tt_environment()
    import torch_xla

    return torch_xla.device()


def tt_sync(wait: bool = True) -> None:
    r"""Execute the pending XLA graph on the Tenstorrent device."""
    import torch_xla

    torch_xla.sync(wait=wait)


def get_tt_memory_info() -> tuple[int, int]:
    r"""Return the (used, total) device memory in bytes, or (0, -1) when it is unavailable.

    The query never opens the device: processes such as the web UI must not initialize the runtime, and
    pjrt-plugin-tt 1.5.0 reports UNIMPLEMENTED for the memory query anyway.
    """
    if not _tt_environment_ready:
        return 0, -1

    try:
        import torch_xla
        import torch_xla.core.xla_model as xm

        info = xm.get_memory_info(torch_xla.device())
        return int(info.get("bytes_used", 0)), int(info.get("bytes_limit", -1))
    except Exception:
        return 0, -1


def get_tt_compile_count() -> int:
    r"""Return the number of XLA graph compilations so far in this process."""
    import torch_xla.debug.metrics as met

    return int(met.counter_value("UncachedCompile") or 0)


def unwrap_optimizer(optimizer: Any) -> "torch.optim.Optimizer":
    while not isinstance(optimizer, torch.optim.Optimizer) and hasattr(optimizer, "optimizer"):
        optimizer = optimizer.optimizer

    return optimizer


class TTAdamW(torch.optim.Optimizer):
    r"""AdamW whose whole update, including the step counter and learning rate, lives on the device.

    `torch.optim.AdamW(capturable=True)` divides by the learning rate inside its update, which produces NaN
    parameters for the zero learning rate of a warmup step. This implementation uses the textbook update
    so any schedule is safe, keeps every scalar as a device tensor and never reads values back to the host.
    """

    def __init__(
        self,
        params: Any,
        lr: Any = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    def materialize_state(self) -> int:
        count = 0
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad and not self.state[param]:
                    self._init_state(param)
                    count += 1

        return count

    def _init_state(self, param: "torch.Tensor") -> None:
        state = self.state[param]
        state["step"] = torch.zeros((), dtype=torch.float32, device=param.device)
        state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional["torch.Tensor"]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, weight_decay = group["lr"], group["eps"], group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                if not state:
                    self._init_state(param)

                state["step"] += 1
                step, exp_avg, exp_avg_sq = state["step"], state["exp_avg"], state["exp_avg_sq"]
                exp_avg.lerp_(param.grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(param.grad, param.grad, value=1 - beta2)
                bias_correction1 = 1 - torch.pow(beta1, step)
                bias_correction2 = 1 - torch.pow(beta2, step)
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
                update = exp_avg / denom / bias_correction1
                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)

                param.sub_(lr * update)

        return loss


def materialize_adamw_state(optimizer: Any) -> int:
    r"""Allocate the AdamW moments on the device before the first step so every step compiles to the same graph."""
    optimizer = unwrap_optimizer(optimizer)
    if isinstance(optimizer, TTAdamW):
        return optimizer.materialize_state()

    if not isinstance(optimizer, torch.optim.AdamW):
        return 0

    count = 0
    for group in optimizer.param_groups:
        for param in group["params"]:
            if not param.requires_grad or optimizer.state[param]:
                continue

            state = optimizer.state[param]
            state["step"] = torch.zeros((), dtype=torch.float32, device=param.device)
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            if group.get("amsgrad", False):
                state["max_exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

            count += 1

    return count


def create_tt_optimizer(model: "torch.nn.Module", training_args: "TrainingArguments") -> "torch.optim.Optimizer":
    r"""Create an AdamW whose step counter and learning rate live on the device.

    Keeping every optimizer scalar on the device makes the compiled graph independent of the step and of
    the scheduler, so the fused forward-backward-update graph compiles once.
    """
    decay_parameters = [name for name in get_parameter_names(model, ALL_LAYERNORM_LAYERS) if "bias" not in name]
    trainable_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    param_groups = [
        {
            "params": [param for name, param in trainable_params if name in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [param for name, param in trainable_params if name not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    param_groups = [group for group in param_groups if group["params"]]
    if not param_groups:
        raise ValueError("No trainable parameters were found for the Tenstorrent optimizer.")

    device = param_groups[0]["params"][0].device
    optimizer = TTAdamW(
        param_groups,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )
    for group in optimizer.param_groups:
        group["lr"] = torch.tensor(float(training_args.learning_rate), dtype=torch.float32, device=device)

    count = optimizer.materialize_state()
    logger.info_rank0(f"Created the Tenstorrent AdamW and pre-allocated the state of {count} trainable tensors.")
    return optimizer


class TTLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    r"""Drive a standard scheduler on host-side floats and stream each new learning rate to the device.

    Writing a Python float into a device tensor bakes it into the compiled graph and forces a recompile on
    every step. Copying it from a freshly transferred one-element tensor keeps the graph static.
    """

    def __init__(self, optimizer: "torch.optim.Optimizer", inner: "torch.optim.lr_scheduler.LRScheduler") -> None:
        self.optimizer = optimizer
        self.inner = inner
        self._push_learning_rates()

    def step(self, *args, **kwargs) -> None:
        self.inner.step(*args, **kwargs)
        self._push_learning_rates()

    def get_last_lr(self) -> list[float]:
        return self.inner.get_last_lr()

    def get_lr(self) -> list[float]:
        return self.inner.get_lr()

    def state_dict(self) -> dict[str, Any]:
        return self.inner.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.inner.load_state_dict(state_dict)
        self._push_learning_rates()

    @property
    def last_epoch(self) -> int:
        return self.inner.last_epoch

    @last_epoch.setter
    def last_epoch(self, value: int) -> None:
        self.inner.last_epoch = value

    @property
    def base_lrs(self) -> list[float]:
        return self.inner.base_lrs

    @property
    def _step_count(self) -> int:
        return self.inner._step_count

    @_step_count.setter
    def _step_count(self, value: int) -> None:
        self.inner._step_count = value

    def _push_learning_rates(self) -> None:
        for group, learning_rate in zip(unwrap_optimizer(self.optimizer).param_groups, self.inner.get_last_lr()):
            current = group["lr"]
            if torch.is_tensor(current):
                source = torch.tensor([float(learning_rate)], dtype=current.dtype).to(current.device)
                current.copy_(source[0])
            else:
                group["lr"] = learning_rate


def create_tt_scheduler(
    training_args: "TrainingArguments", num_training_steps: int, optimizer: Any
) -> "TTLRScheduler":
    r"""Build the configured Hugging Face scheduler on a shadow optimizer and wrap it for the device."""
    base_optimizer = unwrap_optimizer(optimizer)
    shadow_groups = []
    for group in base_optimizer.param_groups:
        learning_rate = group["lr"].item() if torch.is_tensor(group["lr"]) else group["lr"]
        shadow_groups.append({"params": [torch.zeros(1)], "lr": float(learning_rate)})

    shadow_optimizer = torch.optim.SGD(shadow_groups, lr=shadow_groups[0]["lr"])
    shadow_optimizer._opt_called = True
    inner = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=shadow_optimizer,
        num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs=training_args.lr_scheduler_kwargs,
    )
    return TTLRScheduler(base_optimizer, inner)


def get_tt_collator_kwargs(data_args: "DataArguments") -> dict[str, Any]:
    r"""Pad every batch to `cutoff_len` so the compiled graph sees one static shape."""
    return {"padding": "max_length", "max_length": data_args.cutoff_len}


def prepare_tt_training_args(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    r"""Validate and adjust the parsed arguments for single-process training on a Tenstorrent device."""
    if finetuning_args.stage != "sft":
        raise NotImplementedError("Tenstorrent training currently supports `stage: sft` only.")

    if data_args.packing or data_args.neat_packing:
        raise ValueError("Sequence packing is not supported on Tenstorrent devices, rely on `cutoff_len` padding.")

    if training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` is not supported on Tenstorrent devices.")

    if training_args.deepspeed is not None or training_args.fsdp:
        raise ValueError("DeepSpeed and FSDP are not supported on Tenstorrent devices.")

    if training_args.torch_compile:
        raise ValueError("`torch_compile` is not supported on Tenstorrent devices, XLA already compiles the graph.")

    if model_args.quantization_bit is not None:
        raise ValueError("Quantized training is not supported on Tenstorrent devices.")

    if training_args.bf16 or training_args.fp16:
        logger.warning_rank0(
            "Autocast is not used on Tenstorrent devices. The model runs in its compute dtype "
            f"({model_args.compute_dtype}); set `pure_bf16: true` to keep the trainable parameters in bfloat16."
        )
        training_args.bf16 = False
        training_args.fp16 = False
        os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

    if not model_args.disable_gradient_checkpointing:
        logger.info_rank0("Disabling gradient checkpointing: XLA compiles the whole step and recomputation is elided.")
        model_args.disable_gradient_checkpointing = True

    if not training_args.dataloader_drop_last:
        logger.info_rank0("Setting `dataloader_drop_last=True` so every batch has the same static shape.")
        training_args.dataloader_drop_last = True

    if finetuning_args.lora_dropout > 0:
        logger.warning_rank0("`lora_dropout` adds random ops to the device graph; set it to 0 if compilation fails.")

    training_args.dataloader_pin_memory = False
    training_args.average_tokens_across_devices = False
    model_args.device_map = None


class TTTrainingCallback(TrainerCallback):
    r"""Keep the lazy XLA graph bounded to one training step and report compilations."""

    def __init__(self) -> None:
        self._last_compile_count = 0

    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        tt_sync()
        used, total = get_tt_memory_info()
        if total > 0:
            logger.info_rank0(
                f"Tenstorrent device memory before training: {used / 2**30:.2f} / {total / 2**30:.2f} GiB."
            )

    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        tt_sync()

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        tt_sync()

    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        compile_count = get_tt_compile_count()
        if compile_count != self._last_compile_count:
            logger.info_rank0(
                f"Tenstorrent compiled {compile_count - self._last_compile_count} new graph(s) "
                f"since the last log ({compile_count} in total)."
            )
            self._last_compile_count = compile_count
