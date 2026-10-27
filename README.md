# FineTune_LLM

Parameter-efficient fine-tuning (QLoRA) of a small chat LLM on research-paper summarization, with a side-by-side comparison of the base and fine-tuned models.

## What the notebook does

`FineTune_LLM.ipynb` (written for Google Colab with a GPU runtime):

| Step | Detail |
|---|---|
| Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, loaded in 4-bit (NF4, double quantization, fp16 compute) via `bitsandbytes` |
| Dataset | `ccdv/arxiv-summarization`, first 3000 training examples |
| Prompt format | `### Instruction: Summarize the following research:` + first 300 characters of the article + `### Response:` + the paper's abstract |
| Method | LoRA through `peft`: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, target modules `q_proj` and `v_proj`, after `prepare_model_for_kbit_training` |
| Training | HuggingFace `Trainer`, 3 epochs, batch size 1, max sequence length 256, fp16 |
| Output | Adapter and tokenizer saved to `./finetuned_model` |
| Comparison | Base vs fine-tuned `text-generation` pipelines on a sample prompt, then ROUGE and BLEU against the reference abstract (greedy decoding) |
| Chatbot | A simple interactive loop querying the fine-tuned model |

## Run on Colab

1. Open `FineTune_LLM.ipynb` in [Google Colab](https://colab.research.google.com/) (File > Upload notebook, or open it from GitHub).
2. Set the runtime to GPU (Runtime > Change runtime type > T4 GPU).
3. Run all cells in order. The first cell installs `transformers`, `datasets`, `peft`, `accelerate`, `bitsandbytes`, `rouge-score` and `nltk`.
4. The last cell starts an interactive prompt; type `exit` to stop.

A Hugging Face token is not required, though setting `HF_TOKEN` as a Colab secret gives higher download rate limits.

## Results

The notebook keeps its own outputs (training loss logs, base vs fine-tuned generations, ROUGE/BLEU scores); open it to see them.
