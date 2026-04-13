# Multimodal Attack Assessment Pipeline

A three-stage HarmBench-style pipeline for evaluating multimodal jailbreak
attacks on vision-language models (VLMs).

```
Stage 1  generate_test_cases.py   →  test_cases.json + images/
Stage 2  generate_completions.py  →  completions.json
Stage 3  (external HarmBench-style evaluator consumes completions.json)
```

---

## Directory layout

```
pipeline/
├── generate_test_cases.py      # Stage 1 script
├── generate_completions.py     # Stage 2 script
├── behaviors/
│   └── harmbench_behaviors_multimodal.csv   # sample behaviors
├── methods/
│   ├── __init__.py             # method registry
│   ├── base.py                 # BaseMethod (abstract)
│   └── image_hijack.py         # example: pairs text with adversarial image
└── model_adapters/
    ├── __init__.py             # adapter registry
    ├── base.py                 # BaseModelAdapter (abstract)
    ├── llava.py                # LLaVA-1.5 / LLaVA-1.6
    ├── qwen_vl.py              # Qwen-VL-Chat / Qwen2-VL
    ├── minigpt4.py             # MiniGPT-4
    └── openai_vision.py        # GPT-4V / GPT-4o via OpenAI API
```

---

## Quick start

### Stage 1 – generate test cases

```bash
cd <repo_root>
python pipeline/generate_test_cases.py \
    --method image_hijack \
    --behaviors_file pipeline/behaviors/harmbench_behaviors_multimodal.csv \
    --save_dir runs/my_run
```

This produces:
* `runs/my_run/test_cases.json` – JSON manifest
* `runs/my_run/images/` – all referenced image files

### Stage 2 – generate completions

```bash
python pipeline/generate_completions.py \
    --model llava \
    --test_cases_path runs/my_run/test_cases.json \
    --save_path runs/my_run/completions.json \
    --device cuda:0
```

Supported `--model` values:

| Value           | Description                             |
|-----------------|-----------------------------------------|
| `llava`         | LLaVA-1.5 (HuggingFace)                |
| `llava1.6`      | LLaVA-1.6 / LLaVA-NeXT                 |
| `qwen_vl`       | Qwen-VL-Chat (add `--use_qwen2` for Qwen2-VL) |
| `minigpt4`      | MiniGPT-4                               |
| `openai_vision` | GPT-4V / GPT-4o (requires API key)     |

### Stage 3 – evaluate (HarmBench-style)

`completions.json` has the structure expected by HarmBench evaluators:

```json
{
  "<behavior_id>": [
    { "test_case": { "text": "...", "image": "images/...", ... },
      "generation": "<model output>" },
    ...
  ]
}
```

---

## Test-case schema

Each entry in `test_cases.json` follows this schema:

```json
{
  "text": "<string>",
  "image": "images/<filename>",
  "image_source": "<optional original path>",
  "meta": { "behavior_id": "...", "method": "..." }
}
```

**Required fields:** `text`, `image`

**Constraints on `image`:**
* Must be a *relative* path (not absolute).
* Should start with `images/`.
* The referenced file must exist under `{save_dir}/`.

---

## Adding a custom attack method

1. Create `pipeline/methods/my_method.py`.
2. Define a class called `AttackMethod` that inherits from
   `methods.base.BaseMethod`.
3. Implement `generate_test_cases(behaviors) -> dict`.
4. Use it with `--method my_method`.

## Adding a custom model adapter

1. Create `pipeline/model_adapters/my_model.py`.
2. Define a class inheriting from `model_adapters.base.BaseModelAdapter`.
3. Implement `generate(text, image_path) -> str`.
4. Register it in `model_adapters/__init__.py`.

---

## Settings

Model paths are read from `settings/settings.yaml`.  Update that file
with the local paths to each model checkpoint before running Stage 2.

OpenAI credentials: set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`)
as environment variables, or create `openai_env_var.json` in the repo root:

```json
{ "api_key": "sk-...", "url": "https://api.openai.com/v1" }
```
