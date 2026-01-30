# Abraham Lincoln Fine-Tuned Qwen1.5-0.5B Model

## Model Information
- Base Model: Qwen/Qwen1.5-0.5B-Chat
- Fine-tuning Method: LoRA (Low-Rank Adaptation)
- Training Data: Abraham Lincoln documents (1993 examples)
- Training Duration: 6 epochs

## Files
- `adapter_config.json`: LoRA configuration
- `adapter_model.bin`: LoRA weights
- `special_tokens_map.json`: Tokenizer special tokens
- `tokenizer_config.json`: Tokenizer configuration
- `tokenizer.json`: Tokenizer data

## Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"
LORA_PATH = "/Users/shailjatripathi/Desktop/llm_lincoln_project/llm_integration/qwen_0_5b_lora"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float32
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()
```

## Fine-tuning Details
- LoRA Rank (r): 8
- LoRA Alpha: 16
- Target Modules: q_proj, k_proj, v_proj, o_proj
- Learning Rate: 0.0002
- Batch Size: 1
- Max Length: 512
- Epochs: 6

Generated on: 2026-01-06 10:55:56
