# Multilingual Sentiment Analysis Using LoRA and LLaMA 3.1-8B

Fine-tuning large language models (LLaMA 3.1-8B-Instruct) for sentiment classification across 13 Indian languages using efficient low-rank adaptation (LoRA).

## Introduction

This project addresses the challenge of performing sentiment analysis in a multilingual setting, specifically across 13 diverse Indian languages. Traditional models often struggle with generalizing sentiment detection across languages, especially with limited resources.

We utilize the powerful LLaMA 3.1-8B Instruct model and fine-tune it using LoRA (Low-Rank Adaptation) to enable efficient, resource-friendly training while maintaining high performance on sentiment prediction tasks.

## Objective

- Build a multilingual sentiment classifier capable of identifying sentiment polarity (positive, negative, neutral) in user-generated text.
- Focus on resource efficiency and scalability using LoRA without compromising accuracy.
- Enable real-world NLP applications in regional Indian contexts (social media, customer feedback, etc.).

## Key Features

- Supports 13 Indian languages.
- Efficient fine-tuning using LoRA on the LLaMA 3.1-8B model.
- Integrated with Hugging Face Transformers and Datasets.
- Supports easy evaluation and inference generation.
- Designed for real-world low-resource multilingual applications.

## Approach

1. **Model**: Used Meta’s LLaMA 3.1-8B Instruct variant as the base model.
2. **LoRA Fine-Tuning**:
   - Injected low-rank matrices into key transformer layers.
   - Reduced training time and memory footprint.
3. **Dataset**:
   - Multilingual sentiment dataset spanning 13 Indian languages.
   - Preprocessing: Tokenization, text normalization using Hugging Face tokenizers.
4. **Training**:
   - Used PEFT (Parameter-Efficient Fine-Tuning) with LoRA.
   - Leveraged Hugging Face’s Trainer for seamless training and evaluation.
5. **Evaluation**:
   - Accuracy, F1 Score per language.
   - Comparative analysis against zero-shot baseline.

## Results

| Language      | Accuracy | F1 Score |
|---------------|----------|----------|
| Hindi         | 89.4%    | 88.7%    |
| Tamil         | 86.7%    | 86.2%    |
| Telugu        | 88.0%    | 87.4%    |
| Bengali       | 87.6%    | 87.0%    |
| Others (avg.) | 85%+     | 84%+     |

## References

- [LLaMA 3.1-8B by Meta](https://www.llama.com/llama-downloads/)
- Hugging Face Transformers: https://huggingface.co/transformers/
- LoRA: Hu et al., 2021, "LoRA: Low-Rank Adaptation of Large Language Models"
