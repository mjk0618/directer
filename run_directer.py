# Run DIRECTER on a single prompt split into a task part and an instruction part.
#
# Usage:
#     python run_directer.py
#     python run_directer.py --model meta-llama/Llama-3.1-8B-Instruct --max_new_tokens 512
import argparse

from src import (
    DirecterGenerator,
    load_model_and_tokenizer,
    prepare_prompt,
    get_indices_to_scale,
)

# Example: task + instruction (separated)
TASK = "Your task is to write an itinerary for a trip to Japan in a Shakespearean style."
INSTRUCTION = "- Do not use any commas in your response."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--key_scale_factor", type=float, default=100.0, help="alpha")
    parser.add_argument("--plausibility", type=float, default=0.5, help="beta")
    parser.add_argument("--fast_decoding", action="store_true",
                        help="Use a fast attention implementation for decoding after the "
                             "one-time eager layer ranking. Off by default.")
    parser.add_argument("--decoding_attn_implementation", type=str, default="sdpa",
                        help="Attention implementation for decoding when --fast_decoding is set.")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)

    # Build the full prompt and locate the instruction span to steer.
    query = f"{TASK}\n\n{INSTRUCTION}"
    prompt = prepare_prompt(query, tokenizer)
    start_idx, end_idx = get_indices_to_scale(prompt, INSTRUCTION, tokenizer)

    generator = DirecterGenerator(
        model,
        tokenizer,
        key_scale_factor=args.key_scale_factor,
        plausibility=args.plausibility,
        max_new_tokens=args.max_new_tokens,
        fast_decoding=args.fast_decoding,
        decoding_attn_implementation=args.decoding_attn_implementation,
    )
    generator.configure(start_idx, end_idx)
    response = generator.generate(prompt)

    print("=" * 70)
    print("PROMPT:\n" + query)
    print("=" * 70)
    print("DIRECTER RESPONSE:\n" + response)
    print("=" * 70)


if __name__ == "__main__":
    main()
