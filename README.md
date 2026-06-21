# NLP Failure Analysis: Backhanded Compliments

## Objective
This project explores the failure modes of Large Language Models (LLMs) specifically regarding **Pragmatic Inference**. We demonstrate that models like Llama-3.1-8B exhibit a **Positive Lexical Bias**, where they prioritize positive adjectives (e.g., "intelligent", "competent") over the condescending syntactic qualifiers that render a sentence an insult.

## Hypothesis
The model performs a "sentiment sum" rather than a logical parse. It fails to identify that a compliment relative to a low baseline (e.g., "more competent than I expected") is socially negative.


## Setup
1. Clone the repository.
2. Install dependencies:
``` pip install -r requirements.txt ```

    Authenticate with Hugging Face and use personal access token (Required for Llama access):
``` huggingface-cli login ```
3. Run the analysis:
``` python main.py ```

## Methodology

The script uses a zero-shot prompt forcing a single-word classification. It extracts the raw logits from the model's first generated token to calculate a confidence score, allowing to distinguish between "confused" errors and "systemic" errors.