# Fine-tuning on Tenstorrent devices

LlamaFactory can run LoRA supervised fine-tuning (SFT) on Tenstorrent Blackhole and Wormhole
accelerators through PyTorch/XLA and the Tenstorrent PJRT (Portable JAX Runtime) plugin
`pjrt-plugin-tt` from the [tt-xla](https://github.com/tenstorrent/tt-xla) project.

The backend is enabled by the environment variable `PJRT_DEVICE=TT` (or `USE_TT=1`). Nothing else
in LlamaFactory changes: the same YAML configs, datasets, templates and `llamafactory-cli train`
entry point are used.

## Requirements

- Python 3.12 (the `pjrt-plugin-tt` wheel is built for `cp312` only).
- A Tenstorrent device with the kernel-mode driver (TT-KMD) and firmware installed, see the
  [Tenstorrent software installation guide](https://docs.tenstorrent.com/getting-started/README.html).
- The plugin pins `torch==2.11.0+cpu` and its own `torch_xla` build. GPU builds of torch are not used.

## Installation

```bash
git clone https://github.com/hiyouga/LlamaFactory && cd LlamaFactory
uv venv --python 3.12 .venv
printf 'pandas>=2.2,<3.0\nwebsockets>=13.0,<16.0\n' > /tmp/llamafactory-tt-overrides.txt
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --override /tmp/llamafactory-tt-overrides.txt \
  --extra-index-url https://pypi.eng.aws.tenstorrent.com \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "pjrt-plugin-tt==1.5.0.dev20260831000501" \
  "torch @ https://download.pytorch.org/whl/cpu/torch-2.11.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl" \
  "torch-xla @ https://pypi.eng.aws.tenstorrent.com/torch-xla/torch_xla-2.9.0%2Bgited8a445-cp312-cp312-linux_x86_64.whl" \
  "tt_lang @ https://pypi.eng.aws.tenstorrent.com/2026-07/tt-lang/tt_lang-1.1.5.dev20260704%2Blight-cp312-cp312-manylinux_2_34_x86_64.whl" \
  "transformers==4.57.1" "torchvision==0.26.0" "torchaudio==2.11.0" -e .
source .venv/bin/activate
```

Notes on the command:

- The plugin declares `torch`, `torch-xla` and `tt_lang` as direct URL dependencies. `uv` requires
  those URLs to be repeated on the command line, so they are listed explicitly. The URLs come from the
  plugin's own metadata (`pip show -f pjrt-plugin-tt` prints the `Requires-Dist` lines).
- The plugin pins `pandas==3.0.0` and `websockets==16.0`, which conflict with `gradio` (a LlamaFactory
  dependency). The overrides file relaxes these two pins. Both packages are only used by the plugin's
  optional tooling.
- `transformers==4.57.1` is the version validated by Tenstorrent's own training recipes
  ([tt-blacksmith](https://github.com/tenstorrent/tt-blacksmith)).
- Check the [pjrt-plugin-tt index](https://pypi.eng.aws.tenstorrent.com/pjrt-plugin-tt/) for newer
  nightly builds. When you change the plugin version, update the `torch-xla` URL to the one listed in
  the new wheel's metadata.

## Running

```bash
PJRT_DEVICE=TT TT_VISIBLE_DEVICES=0 llamafactory-cli train examples/tenstorrent/llama3_lora_sft.yaml
```

or use the wrapper script, which sets `PJRT_DEVICE` and requires `TT_VISIBLE_DEVICES`:

```bash
TT_VISIBLE_DEVICES=1 examples/tenstorrent/run_llama3_lora_sft.sh
```

Pick a chip that no other process uses. tt-metal takes a per-chip lock; when the chip is busy the run
blocks with `Waiting for lock 'CHIP_IN_USE_<n>_PCIe' ... held by PID <pid>` in the log instead of
failing. `tt-smi -ls` lists the chip indices; on a two-chip `p300` board each chip is one index.

The output adapter is a standard PEFT LoRA checkpoint. To serve it with an inference stack that does not
load adapters (for example tt-inference-server), merge it into the base weights first:

```bash
llamafactory-cli export --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --adapter_name_or_path saves/llama3.1-8b-instruct/lora/sft-tt --template llama3 \
  --finetuning_type lora --export_dir models/llama3.1-8b-instruct-sft --export_device cpu
```

Environment variables:

| Variable | Meaning |
| --- | --- |
| `PJRT_DEVICE=TT` or `USE_TT=1` | Enable the Tenstorrent backend. |
| `TT_VISIBLE_DEVICES` | Comma-separated chip indices (as listed by `tt-smi -ls`) or PCI bus IDs to use. Training is single-process and uses the first visible chip. Leave unset to open every chip in the host. |
| `TT_MESH_GRAPH_DESC_PATH` | Fabric mesh graph descriptor. When exactly one Blackhole chip is visible, LlamaFactory sets this to the single-chip (`p150`) descriptor shipped with the plugin, which is required to use one chip of a two-chip `p300` board. |
| `TT_METAL_LOGS_PATH` | Where tt-metal writes its `generated/` diagnostics. Defaults to `~/.cache/llamafactory/tenstorrent`. |
| `LLAMAFACTORY_TT_COMPILE_OPTIONS` | JSON dictionary merged into the tt-mlir compile options. Defaults: `{"fp32_dest_acc_en": true, "math_fidelity": "hifi4", "enable_const_eval": false}`. Const-eval caches transformed copies of the frozen weights on the device; with an 8B model that second copy does not fit in the 32 GB of one chip. |
| `TTXLA_LOGGER_LEVEL=DEBUG` | Verbose compiler logs from tt-xla, useful when a graph fails to compile. |

## Web UI (LlamaBoard)

```bash
TT_VISIBLE_DEVICES=1 examples/tenstorrent/run_webui.sh
```

The script puts the virtual environment on `PATH` (the UI starts training as a `llamafactory-cli`
subprocess that inherits the UI's environment), sets `PJRT_DEVICE=TT` and listens on port 7860. The UI
process itself never opens the chip.

A ready-made form lives in `examples/tenstorrent/llamaboard/`. Install it once:

```bash
mkdir -p llamaboard_config llamaboard_cache
cp examples/tenstorrent/llamaboard/tenstorrent_llama3_lora_sft.yaml llamaboard_config/
cp examples/tenstorrent/llamaboard/user_config.yaml llamaboard_cache/
```

`user_config.yaml` preselects `Llama-3.1-8B-Instruct` and maps it to the current Hugging Face repo name
`meta-llama/Llama-3.1-8B-Instruct` (LlamaFactory's built-in entry still points at the old
`Meta-Llama-3.1-8B-Instruct` name, which does not resolve offline). In the Train tab pick
`tenstorrent_llama3_lora_sft.yaml` under **Config path**, click **Load arguments**, check that the
cutoff length is 128 and the compute type is `pure_bf16`, then **Start**. Starting with the form defaults
(cutoff length 2048, all LoRA modules) runs out of device memory.

## What the backend does

PyTorch/XLA traces each training step into one graph that tt-mlir compiles for the device. Every
distinct graph is compiled once (this takes minutes for an 8B model) and then replayed. LlamaFactory
therefore adjusts the training setup so the graph is identical on every step:

- Batches are padded to `cutoff_len` (`padding="max_length"`) and `dataloader_drop_last` is forced on.
- The optimizer is a capturable `AdamW` with a device-resident learning rate. Learning-rate schedulers
  work; the new value is streamed to the device each step instead of being baked into the graph.
- The AdamW moments are allocated before the first step.
- The graph is executed after every optimizer step (and after every gradient-accumulation sub-step).
- Autocast (`bf16: true`) is turned off; the model runs in its compute dtype. Use `pure_bf16: true` to
  keep the LoRA weights in bfloat16 as well, which is the configuration validated by Tenstorrent.
- Gradient checkpointing is disabled because XLA compiles the whole step and elides recomputation.
- Sequence packing, `predict_with_generate`, quantization, DeepSpeed, FSDP and `torch_compile` are
  rejected with an error. Only `stage: sft` is supported for now.

The log line `Tenstorrent compiled N new graph(s)` at each logging step reports recompilations. After the
first two or three steps this number should stay at zero; if it keeps growing, some input shape or
Python scalar changes between steps.

## Measured on one Blackhole chip (p300, 32 GB)

Llama 3.1 8B Instruct, LoRA rank 8 on `q_proj`/`v_proj`, `cutoff_len: 128`, `pure_bf16`,
pjrt-plugin-tt 1.5.0.dev20260831000501, measured 2026 Sep 30:

| Setting | Result |
| --- | --- |
| `per_device_train_batch_size: 8` or `4` | Out of device DRAM during the first forward-backward pass (the 15 GB of weights plus the per-step intermediates exceed 32 GB). |
| `per_device_train_batch_size: 2`, `gradient_accumulation_steps: 4` | Trains. First optimizer step about 3.5 minutes (graph compilation), then about 3.3 s per optimizer step of 8 samples. One epoch of `identity,alpaca_en_demo` (137 steps) took 655 s, training loss went from 1.40 to about 1.0, 19 distinct graphs were compiled (17 at step 1, one at step 2, one at the final save). The saved adapter loads with PEFT on the CPU and changes the model's answers. |

Raise the effective batch through `gradient_accumulation_steps`, not through the per-device batch size.
Longer `cutoff_len` needs a smaller per-device batch.

## Multi-chip

The current integration is single chip. tt-xla supports SPMD tensor and data parallelism across chips,
see the mesh configurations in tt-blacksmith for the parameters involved.
