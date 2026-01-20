# Refusal Vector Ablation for Lily-Cybersecurity-7B model

### [View Model on Hugging Face](https://huggingface.co/NiroshanDb23/Lily-Cybersecurity-7B-Uncensored-GGUF)

## Overview
This project implements **Refusal Vector Ablation** often called "Abliteration" on the `Lily-Cybersecurity-7B-v0.2` model. 

The goal was to remove the model's safety refusal mechanisms without damaging its general reasoning capabilities, specifically to restore its ability to assist in **Ethical Hacking** and **Penetration Testing** tasks where standard safety guardrails often trigger false positives.

## Methodology
The script utilizes **activation steering** to surgically modify the model's weights, 

- **Refusal Isolation** I run the model against opposing pairs of prompts (x vs. y) and capture the residual stream activations at the final token of Layer 14.
- **Vector Computation** I compute the mean difference between these activations to isolate a "Refusal Direction" vector ($\hat{r}$).
- **Weight Orthogonalization** in project the refusal vector onto the MLP Down-Projection weights ($W_{down}$) of every layer and subtract it.
    
    $$W_{new} = W - (W \cdot \hat{r}) \hat{r}^T$$

This effectively removes the model's ability to represent the "refusal" concept in its output features while preserving other knowledge.

## Tech Stack
* **Python 3.10+**
* **PyTorch** (Linear algebra & tensor manipulation)
* **Transformers** (Model loading & hooking)
* **Google Colab** (T4 GPU Runtime)

## Try it

- **Install Dependencies**
   ```bash
   pip install -r requirements.txt

- **Run the Ablation Script**
    ```bash
   python abliterate.py

- The script will save the modified model to ./Lily-Cybersecurity-Abliterated

## Disclaimer

- This model is intended for cybersecurity research and authorized penetration testing. The removal of safety guardrails requires the user to exercise responsibility. The author is not liable for misuse.

