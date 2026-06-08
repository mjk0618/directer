# DIRECTER: Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection

[![Paper](https://img.shields.io/badge/Paper-Openreview-b31b1b.svg)](https://openreview.net/forum?id=OpuPBNcQwe)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://mjk0618.github.io/directer/)

![DIRECTER Method Overview](./assets/directer.png)

## Overview

**DIRECTER** is a novel inference-time activation steering method designed to significantly improve how Large Language Models (LLMs) follow complex instructions while mitigating the common risk of "oversteering."

While activation steering techniques can effectively force models to adhere to constraints, they often suffer from a trade-off: excessive emphasis on the instruction can degrade the overall coherence and quality of the generated text. **DIRECTER** solves this by dynamically modulating steering strength at every decoding step.

### Key Mechanism
DIRECTER couples KV cache steering with a **plausibility-guided decoding loop**. At each step, the method:
1.  **Steers:** Tentatively amplifies the "Key" vectors in the KV cache associated with the instruction.
2.  **Checks Plausibility:** Compares the steered output distribution against the raw model's distribution.
3.  **Modulates:** If the steered output is deemed implausible (deviates too far from the model's natural distribution), DIRECTER progressively reduces the steering strength by removing layers from the intervention set.

This process is guided by a lightweight, one-time **Sensitivity Analysis** that ranks layers based on their influence, ensuring that the most effective layers are prioritized.

## Installation

```bash
conda create -n directer python=3.10 -y
conda activate directer
pip install -r requirements.txt
```

## Usage

Run DIRECTER on a prompt whose task and instruction parts are separated:

```bash
python run_directer.py
```

The example run uses the following task and instruction:

- **Task:** *Write an itinerary for a trip to Japan in a Shakespearean style.*
- **Instruction:** *Do not use any commas in your response.*

You can change the model or generation budget from the command line:

```bash
python run_directer.py --model meta-llama/Llama-3.1-8B-Instruct --max_new_tokens 2048
```

Because DIRECTER steers the KV cache rather than the attention kernel, the
autoregressive decoding stays compatible with fast attention implementations.
The one-time layer ranking uses eager attention; the subsequent decoding can
optionally run under a fast kernel (e.g. SDPA) via `--fast_decoding` (off by
default):

```bash
python run_directer.py --fast_decoding --decoding_attn_implementation sdpa
```

To apply DIRECTER to your own prompt, separate the task and instruction, locate
the instruction token span, and call the generator:

```python
from src import (
    DirecterGenerator,
    load_model_and_tokenizer,
    prepare_prompt,
    get_indices_to_scale,
)

model, tokenizer = load_model_and_tokenizer("meta-llama/Llama-3.1-8B-Instruct")

task = "Your task is to write an itinerary for a trip to Japan in a Shakespearean style."
instruction = "- Do not use any commas in your response."
query = f"{task}\n\n{instruction}"

prompt = prepare_prompt(query, tokenizer)
start_idx, end_idx = get_indices_to_scale(prompt, instruction, tokenizer)

generator = DirecterGenerator(model, tokenizer, key_scale_factor=100.0, plausibility=0.5)
generator.configure(start_idx, end_idx)        # register instruction token span
response = generator.generate(prompt)
print(response)
```

## Repository Layout

```
src/
  cache_utils.py   # KV-cache key scaling on the instruction span
  model_utils.py   # model/tokenizer loading + prompt helpers
  generator.py     # DIRECTER decoding (attention-sensitivity ranking + plausibility loop)
run_directer.py    # example entry point
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{
	kang2026enhancing,
	title={Enhancing Instruction Following of {LLM}s via Activation Steering with Dynamic Rejection},
	author={Minjae Kang and Jaehyung Kim},
	booktitle={The Fourteenth International Conference on Learning Representations},
	year={2026},
	url={https://openreview.net/forum?id=OpuPBNcQwe}
}
```
