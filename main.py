###########################################################
# Exercise 2 - Advanced Natural Language Processing 67664 #
###########################################################
### Imports ###
import torch
import torch.nn.functional as F
import pandas as pd
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
### Imports End ###
### Global Vars ###
# Original examples (first experiment)
EXAMPLES_OLD = [
    "For someone of your social class, you're actually very articulate and "
    "rather pleasant to talk to.",
    "How interesting to have you here. Your attendance fills a much-needed gap "
    "in the guest list.",
    "I'm impressed someone such as yourself can afford something so pricy for "
    "a vehicle.",
    "How refreshing not to be burdened with the responsibility of manners!",
    "You know, you’ve got the perfect face for radio!",
    "You look lovely in that dress, it really helps to hide your figure.",
    "You’re much more competent than I expected from someone in your position.",
    "Surprisingly, you are far more intelligent than you appear.",
    "It's so impressive how you managed to graduate, considering the "
    "neighborhood you grew up in.",
    "I wish I was as smart as you think you are.",
]

# Refined examples (second experiment)
EXAMPLES = [
    "You’re much more competent than I expected from someone in your position.",
    "Surprisingly, you are far more intelligent than you appear.",
    "It’s impressive how polished your presentation was, given your usual "
    "style.",
    "I’m impressed by how you manage to stay so confident despite your obvious "
    "lack of expertise.",
    "It's refreshing to see someone so unbothered by their own lack of "
    "experience.",
    "I’ve always admired how you don't let your lack of natural talent hold you "
    "back.",
    "You are surprisingly well-spoken for someone who didn't attend a "
    "prestigious university.",
    "I love how you don't feel the need to keep up with standard professional "
    "etiquette.",
    "It's so brave of you to share your opinion even when you aren't an expert "
    "on the topic.",
]
### Global Vars End ###

def classify_with_confidence(model, tokenizer, statement):
    """
    Classifies a given statement as Positive, Negative, or Neutral,
    and returns the model's confidence level for the final prediction.
    Args:
        model: Loaded HuggingFace causal language model.
        tokenizer: Corresponding tokenizer.
        statement (str): Input statement to classify.
    Returns:
        str: Prediction label with confidence level.
    """
    prompt = f"""Read the following statement.
Classify the speaker's attitude toward the person being addressed as one of:
Positive
Negative
Neutral
Base your answer on the overall implied attitude, not just the literal meaning.
Respond with exactly one word.
Do not explain.
Statement: "{statement}"
"""
    # Format prompt using chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cpu")
    # Generate output and retain token scores for confidence calculation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,  # Use 3 for Llama, 20 for Qwen
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    # Decode generated label
    response = tokenizer.decode(
        outputs.sequences[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()
    # Confidence calculation based on final generated token
    logits = outputs.scores[-1][0]
    probabilities = F.softmax(logits, dim=-1)
    token_id = outputs.sequences[0][-1]
    confidence_score = probabilities[token_id].item()
    # Convert raw probability into confidence category
    if confidence_score > 0.80:
        confidence_label = "very confident"
    elif confidence_score > 0.50:
        confidence_label = "confident"
    else:
        confidence_label = "unsure"
    result = f"{response} ({confidence_label})"
    print(result)
    return result


def run_model(model_name):
    """
    Loads a model and runs inference on all examples.
    Args:
        model_name (str): HuggingFace model identifier.
    """
    print(f"Loading {model_name} onto RAM (this may take a minute)...")
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"}  # Force CPU inference
    )
    print("Starting inference...")
    results = []
    # Run classification on all examples
    for i, example in enumerate(EXAMPLES, 1):
        start_time = time.time()
        output = classify_with_confidence(model, tokenizer, example)
        elapsed = time.time() - start_time
        print(f"Done {i}/{len(EXAMPLES)} (Took {elapsed:.2f}s)")
        results.append({
            "Index": i,
            "Input": example,
            "Model Output": output
        })
    # Display results as table
    df = pd.DataFrame(results)
    print(f"\n--- Results for {model_name} ---")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_model("meta-llama/Llama-3.1-8B-Instruct")
    # run_model("Qwen/Qwen3-8B")