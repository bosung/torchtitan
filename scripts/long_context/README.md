## Usage Instructions

#### Run on a single GPU.

```bash
NGPU=1 CONFIG_FILE=./train_configs/save_ckpt.toml CHECKPOINT_DIR=./outputs/checkpoint/ \
PROMPT="What is the meaning of life?" \
./scripts/generate/run_llama_generate.sh --max_new_tokens=32 --temperature=0.8 --seed=3
```

#### Run on 8 GPUs and pipe results to a json file.

```bash
NGPU=8 CONFIG_FILE=./train_configs/save_ckpt.toml CHECKPOINT_DIR=./outputs/checkpoint/step-10 \
./scripts/long_contex/run_llama_generate.sh --max_new_tokens=32 --temperature=0.8 --seed=3 --out --ctx_len 8192 > output.json
```

