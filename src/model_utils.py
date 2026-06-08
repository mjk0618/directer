import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_id="meta-llama/Llama-3.1-8B-Instruct", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    # Greedy decoding.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.update(
            temperature=None, top_p=None, top_k=None, do_sample=False
        )

    return model, tokenizer


def prepare_prompt(query: str, tokenizer):
    # Wrap the query in the chat template and tokenize it.
    chat = [{"role": "user", "content": query}]
    return tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        add_special_tokens=False,
        return_tensors="pt",
        return_dict=True,
    )


def get_indices_to_scale(prompt, segment: str, tokenizer, verbose=True):
    # Locate the token span of segment (the instruction) inside the prompt,
    # returning (start, end) token indices, or (-1, -1) if not found.
    prompt_ids = prompt.input_ids.squeeze()
    segment_ids = tokenizer(segment, add_special_tokens=False, return_tensors="pt").input_ids.squeeze()

    for i in range(len(prompt_ids) - len(segment_ids) + 1):
        if torch.equal(prompt_ids[i : i + len(segment_ids)], segment_ids):
            return i, i + len(segment_ids)
    if verbose:
        print(f"[get_indices_to_scale] segment not found in prompt:\n{segment}")
    return -1, -1
