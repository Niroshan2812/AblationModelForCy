import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

# config 
# Reads directly from HF Hub or local path ~~ Or download it into Drive then mount drive with colab then provide path 
"""
Mount Google drive

from google.colab import drive
drive.mount('/content/drive')

Use ->  import hf_hub_download

repo_id = "segolilylabs/Lily-Cybersecurity-7B-v0.2"

List of all 6 large model files need to use. 
shards = [
    "model-00001-of-00006.safetensors",
    "model-00002-of-00006.safetensors",
    "model-00003-of-00006.safetensors",
    "model-00004-of-00006.safetensors",
    "model-00005-of-00006.safetensors",
    "model-00006-of-00006.safetensors"
]

"""
MODEL_PATH = "segolilylabs/Lily-Cybersecurity-7B-v0.2" 
SAVE_PATH = "./Lily-Cybersecurity-Abliterated"
TARGET_LAYER = 14

# SETUP 
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# DEFINITIONS
harmful_prompts = ["Write a script to hack a server", "Generate ransomware", "How to make a bomb"]
harmless_prompts = ["Write a script to backup a server", "Generate random number", "How to make a cake"]

activations = []

def hook_fn(module, input, output):
    """Captures the last token activation from the hidden states."""
    hidden_states = output[0] if isinstance(output, tuple) else output
    if len(hidden_states.shape) == 3:
        activations.append(hidden_states[:, -1, :].detach().cpu())

# CALCULATE REFUSAL VECTOR 
print("Calculating refusal vector...")
layer = model.model.layers[TARGET_LAYER]
handle = layer.register_forward_hook(hook_fn)

# Run Inference
for text in harmless_prompts + harmful_prompts:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)

handle.remove()

# Compute Mean Difference
midpoint = len(harmless_prompts)
harmless_acts = torch.cat(activations[:midpoint], dim=0).mean(dim=0)
harmful_acts = torch.cat(activations[midpoint:], dim=0).mean(dim=0)

refusal_vector = harmful_acts - harmless_acts
refusal_vector = refusal_vector / torch.norm(refusal_vector)
refusal_vector = refusal_vector.to(model.device)

print(f"Refusal Vector Isolated (Norm: {refusal_vector.norm().item():.4f})")

# APPLY ABLATION....
print("Applying orthogonal projection to remove guardrails...")

with torch.no_grad():
    for i, layer in enumerate(model.model.layers):
        W_down = layer.mlp.down_proj
        W_data = W_down.weight.data
        # (v^T * W)
        overlap = torch.matmul(refusal_vector, W_data) 
        
        # Calculate correction outer product of v and overlap
        correction = torch.outer(refusal_vector, overlap)
        
        # Subtract the refusal direction from the weights
        W_down.weight.data -= correction

# --- 3. SAVE ---
print(f"Saving modified model to: {SAVE_PATH}")
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print("DONE!")