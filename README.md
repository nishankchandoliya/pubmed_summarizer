# PubMed abstract summarizer

Small script that cleans our messy PubMed JSON and asks our local Llama 3.3 70B (vLLM on the DGX Spark) to summarize the clinical outcome of each study in one sentence

## Project Demo

[ Code Walkthrough ](https://youtube.com/shorts/bfIITLM2fCc?si=C5W0lVDsDoIruOFt)

## Files

- `summarize_abstracts.py`: the script
- `abstracts.json`: sample raw data from the legacy db
- `requirements.txt`: only needs requests

## How to run

```bash
pip install -r requirements.txt
python summarize_abstracts.py abstracts.json --output results.json
```

If the server url or model name changes, set them with env vars:

```bash
export LLM_API_URL=http://192.168.1.100:8000/v1/chat/completions
export LLM_MODEL=llama-3.3-70b
```

Output goes to `results.json`: one summary per study, plus a `skipped` list with the reason for each entry we dropped

Exit code is 1 if any record failed, so it is easy to catch in cron

## Cleaning

- field names are mixed (`ArticleTitle` vs `title`, `abstract_text` vs `Abstract`), so keys are lower cased and we check every name we know
- ids like `PMID:349281` and `8829102` become plain digits, an id with no number in it (`ERR_NOT_FOUND`) is skipped
- `null`, empty strings and `"Unknown"` count as missing, but only when that is the whole value (the second abstract starts with "Null hypothesis" and it is real data)
- no abstract means nothing to summarize, so that entry is skipped and logged
- duplicate PMIDs are skipped
- very long abstracts are cut to 6000 chars so the prompt stays under the server context limit

## Timeouts and retries

- 5s to connect, 120s to wait for the answer
- timeouts, dropped connections and 429 or 5xx errors get retried, 3 tries in total (wait 5s, then 10s)
- 400 or 404 errors are not retried because they fail the same way again
- if a record still fails it is saved with status `failed` and the script goes on to the next one

## Starting the server on the DGX Spark

```bash
# clear the linux file cache first, on the spark it eats into the same memory the gpu uses
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

docker run -d --name llama70b \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --restart unless-stopped \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/vllm:26.10-py3 \
  vllm serve nvidia/Llama-3.3-70B-Instruct-FP4 \
    --served-model-name llama-3.3-70b \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8
```

Check it is up before running the script (first start downloads around 40 GB so it takes a while):

```bash
docker logs -f llama70b
curl http://192.168.1.100:8000/v1/models
free -h
```

`free -h` is the easy way to see memory here, `nvidia-smi --query-gpu` shows N/A for memory on the Spark

### Why these settings (this is what stops the OOM)

- Llama 3.3 70B in bf16 is around 140 GB of weights and the Spark has 128 GB in total, so the normal model can't even load
- NVIDIA's NVFP4 version `nvidia/Llama-3.3-70B-Instruct-FP4` is around 40 GB and runs on Blackwell, which is what the GB10 chip is (no HF token needed)
- the 128 GB is shared by the CPU and the GPU, and by default vLLM grabs 90% of it up front for weights and KV cache, so Linux and Docker are left with almost nothing and the whole box starts swapping or gets OOM killed
- `--gpu-memory-utilization 0.6` gives vLLM around 75 GB (40 GB weights, the rest is KV cache) and keeps around 50 GB for the system, NVIDIA's own Spark recipe for the 128 GB model uses 0.6 too
- `--max-model-len 8192`: Llama 3.3 defaults to 128k context and one 128k request needs about 40 GB of KV cache in bf16, our prompts are under 2k tokens so 8192 is plenty
- `--kv-cache-dtype fp8` halves the KV cache (160 KB per token instead of 320 KB)
- `--max-num-seqs 4`: at most 4 requests run at once, the rest wait in a queue instead of using more memory
- `--max-num-batched-tokens 8192` limits how many tokens go through in one step, so memory during prefill stays small
- `nvcr.io/nvidia/vllm:26.10-py3` is NVIDIA's vLLM build for ARM plus Blackwell (vLLM 0.31.0, CUDA 13.4), keep the tag pinned and don't use latest
- `--ipc=host` and the two ulimits are the flags NVIDIA's containers ask for, PyTorch needs the shared memory
- `-p 8000:8000` with `--host 0.0.0.0` makes it reachable at 192.168.1.100:8000 on our network
- `--restart unless-stopped` brings the server back if it crashes or the Spark reboots

On the script side: one request at a time, `max_tokens` is 100 and abstracts are cut at 6000 chars, so we never send a huge prompt or a flood of requests

This is for the 128 GB Spark, a 70B model won't fit safely on the 64 GB one so use a smaller model there

## Author

Nishank Chandoliya
