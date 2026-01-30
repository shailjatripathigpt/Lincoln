#!/usr/bin/env python3
"""
Qwen1.5-0.5B LoRA Fine-Tuning for Abraham Lincoln Documents
FIXED VERSION with proper path handling and error checking
"""

import os
import json
import torch
import logging
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import sys
from datetime import datetime

# =====================================================
# SETUP LOGGING
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fine_tuning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =====================================================
# SAFETY SETTINGS
# =====================================================
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =====================================================
# CONFIGURATION - UPDATED WITH ABSOLUTE PATHS
# =====================================================

# Get absolute paths
def get_absolute_paths():
    """Get absolute paths for all directories"""
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.absolute()
    
    # Model configuration
    MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"
    
    # Dataset directory - FIXED PATH
    DATASET_DIR = project_root / "data_processing" / "outputs" / "enhanced_data"
    
    # Output directory - FIXED PATH (in llm_integration folder)
    OUTPUT_DIR = script_dir / "qwen_0_5b_lora"
    
    return MODEL_NAME, DATASET_DIR, OUTPUT_DIR

MODEL_NAME, DATASET_DIR, OUTPUT_DIR = get_absolute_paths()

# Training parameters
MAX_LENGTH = 512  # Reduced for efficiency
BATCH_SIZE = 1
EPOCHS = 6
LEARNING_RATE = 2e-4
GRADIENT_ACCUMULATION_STEPS = 8

# =====================================================
# DATA LOADING - FIXED
# =====================================================

def load_dataset():
    """Load and prepare the Lincoln documents dataset"""
    logger.info(f"📂 Loading dataset from: {DATASET_DIR}")
    
    # Check if dataset directory exists
    if not DATASET_DIR.exists():
        logger.error(f"❌ Dataset directory not found: {DATASET_DIR}")
        logger.error(f"   Current directory: {Path.cwd()}")
        logger.error(f"   Please make sure to run data processing first:")
        logger.error(f"   1. python data_processing/cleaner.py")
        logger.error(f"   2. python data_processing/metadata_extractor.py")
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    
    # List JSON files
    json_files = list(DATASET_DIR.glob("*.json"))
    logger.info(f"📄 Found {len(json_files)} JSON files")
    
    if len(json_files) == 0:
        raise RuntimeError(f"No JSON files found in {DATASET_DIR}")
    
    # Sample some file names
    sample_files = [f.name for f in json_files[:5]]
    logger.info(f"   Sample files: {', '.join(sample_files)}")
    
    texts = []
    processed_count = 0
    error_count = 0
    
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            
            # Extract text from different possible locations
            text = ""
            if "content" in doc and "full_text" in doc["content"]:
                text = doc["content"]["full_text"]
            elif "full_text" in doc:
                text = doc["full_text"]
            elif "text" in doc:
                text = doc["text"]
            
            # Clean and validate text
            if text and len(text.strip()) > 100:
                cleaned_text = text.strip()
                texts.append(cleaned_text)
                processed_count += 1
                
                # Log first 3 samples
                if processed_count <= 3:
                    logger.info(f"📝 Sample {processed_count}: {cleaned_text[:100]}...")
            
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Invalid JSON in {file_path.name}: {e}")
            error_count += 1
        except Exception as e:
            logger.warning(f"⚠️ Error processing {file_path.name}: {e}")
            error_count += 1
    
    logger.info(f"✅ Successfully processed {processed_count} documents")
    if error_count > 0:
        logger.warning(f"⚠️ Failed to process {error_count} documents")
    
    if len(texts) == 0:
        raise RuntimeError("No valid training text found in any documents")
    
    logger.info(f"📚 Total training examples: {len(texts)}")
    logger.info(f"📊 Average text length: {sum(len(t) for t in texts) / len(texts):.0f} characters")
    
    return Dataset.from_dict({"text": texts})

# =====================================================
# MAIN FUNCTION - FIXED
# =====================================================

def main():
    """Main fine-tuning function"""
    print("\n" + "="*80)
    print("🏛️  ABRAHAM LINCOLN - QWEN1.5-0.5B LoRA FINE-TUNING")
    print("="*80)
    
    # Display configuration
    logger.info(f"🔧 Configuration:")
    logger.info(f"   Base Model: {MODEL_NAME}")
    logger.info(f"   Dataset: {DATASET_DIR}")
    logger.info(f"   Output: {OUTPUT_DIR}")
    logger.info(f"   Max Length: {MAX_LENGTH}")
    logger.info(f"   Batch Size: {BATCH_SIZE}")
    logger.info(f"   Epochs: {EPOCHS}")
    logger.info(f"   Learning Rate: {LEARNING_RATE}")
    
    # Check and create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory ready: {OUTPUT_DIR}")
    
    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
        logger.info("🍎 Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        logger.info(f"🖥️ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("💻 Using CPU (training will be slow)")
    
    # =====================================================
    # 1. LOAD TOKENIZER
    # =====================================================
    logger.info("\n📥 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"✅ Tokenizer loaded successfully")
        logger.info(f"   Vocab size: {tokenizer.vocab_size}")
        logger.info(f"   Padding token: {tokenizer.pad_token}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load tokenizer: {e}")
        logger.error("   Make sure you have internet connection to download the model")
        logger.error("   Or check if the model name is correct")
        raise
    
    # =====================================================
    # 2. LOAD BASE MODEL
    # =====================================================
    logger.info("\n📥 Loading base model...")
    try:
        # Set appropriate torch dtype based on device
        if device == "cpu":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16  # Use FP16 for GPU to save memory
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=None,  # We'll handle device placement manually
            low_cpu_mem_usage=True
        )
        
        # Disable cache to save memory
        model.config.use_cache = False
        
        # Move to device
        if device == "mps":
            model = model.to("mps")
        elif device == "cuda":
            model = model.cuda()
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Base model loaded successfully")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Model dtype: {model.dtype}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load base model: {e}")
        logger.error("   This could be due to:")
        logger.error("   1. No internet connection")
        logger.error("   2. Insufficient RAM/VRAM")
        logger.error("   3. Model not available")
        raise
    
    # =====================================================
    # 3. CONFIGURE LoRA
    # =====================================================
    logger.info("\n⚙️ Configuring LoRA adapter...")
    try:
        lora_config = LoraConfig(
            r=8,  # Rank
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",  # Query projection
                "k_proj",  # Key projection  
                "v_proj",  # Value projection
                "o_proj",  # Output projection
            ]
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        logger.info("📊 Trainable parameters:")
        model.print_trainable_parameters()
        
        # Count parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   Trainable: {trainable_params:,} parameters")
        logger.info(f"   Total: {total_params:,} parameters")
        logger.info(f"   Percentage: {(trainable_params/total_params*100):.4f}%")
        
    except Exception as e:
        logger.error(f"❌ Failed to configure LoRA: {e}")
        raise
    
    # =====================================================
    # 4. PREPARE DATASET
    # =====================================================
    logger.info("\n📚 Preparing dataset...")
    try:
        # Load dataset
        dataset = load_dataset()
        
        # Tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH
            )
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset"
        )
        
        logger.info(f"✅ Dataset prepared successfully")
        logger.info(f"   Examples: {len(tokenized_dataset)}")
        logger.info(f"   Input length: {MAX_LENGTH} tokens")
        
        # Show sample tokenized example
        if len(tokenized_dataset) > 0:
            sample = tokenized_dataset[0]
            logger.info(f"   Sample input IDs length: {len(sample['input_ids'])}")
        
    except Exception as e:
        logger.error(f"❌ Failed to prepare dataset: {e}")
        raise
    
    # =====================================================
    # 5. SETUP TRAINING
    # =====================================================
    logger.info("\n⚙️ Setting up training...")
    try:
        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            warmup_steps=100,
            logging_steps=10,
            save_steps=100,
            save_total_limit=3,
            
            # Optimization settings
            fp16=(device != "cpu"),  # Use FP16 for GPU
            bf16=False,  # Don't use bfloat16
            optim="adamw_torch",
            max_grad_norm=1.0,
            
            # Other settings
            report_to="none",
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            dataloader_num_workers=0 if device == "mps" else 2,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal language modeling
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        logger.info("✅ Training setup complete")
        logger.info(f"   Total steps: {len(tokenized_dataset) * EPOCHS / (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS):.0f}")
        logger.info(f"   Checkpoints will be saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"❌ Failed to setup training: {e}")
        raise
    
    # =====================================================
    # 6. START TRAINING
    # =====================================================
    print("\n" + "="*80)
    print("🚀 STARTING FINE-TUNING")
    print("="*80)
    logger.info("Beginning fine-tuning process...")
    logger.info(f"Estimated time: {EPOCHS * len(tokenized_dataset) / (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) * 0.5:.0f} minutes")
    logger.info("(Press Ctrl+C to interrupt)")
    
    try:
        # Train the model
        trainer.train()
        
        logger.info("✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted by user")
        print("\n⚠️ Training was interrupted. Partial results may be saved.")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =====================================================
    # 7. SAVE THE MODEL
    # =====================================================
    logger.info("\n💾 Saving fine-tuned model...")
    try:
        # Save the model
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # List saved files
        saved_files = list(OUTPUT_DIR.glob("*"))
        logger.info(f"✅ Model saved successfully to: {OUTPUT_DIR}")
        logger.info(f"📁 Files created:")
        
        for file in saved_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            logger.info(f"   • {file.name:25} ({size_mb:.2f} MB)")
        
        # Create a README file (FIXED - no f-string with triple quotes)
        readme_lines = [
            "# Abraham Lincoln Fine-Tuned Qwen1.5-0.5B Model",
            "",
            "## Model Information",
            f"- Base Model: Qwen/Qwen1.5-0.5B-Chat",
            f"- Fine-tuning Method: LoRA (Low-Rank Adaptation)",
            f"- Training Data: Abraham Lincoln documents ({len(tokenized_dataset)} examples)",
            f"- Training Duration: {EPOCHS} epochs",
            "",
            "## Files",
            "- `adapter_config.json`: LoRA configuration",
            "- `adapter_model.bin`: LoRA weights",
            "- `special_tokens_map.json`: Tokenizer special tokens",
            "- `tokenizer_config.json`: Tokenizer configuration",
            "- `tokenizer.json`: Tokenizer data",
            "",
            "## Usage",
            "```python",
            "from transformers import AutoTokenizer, AutoModelForCausalLM",
            "from peft import PeftModel",
            "",
            f'MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"',
            f'LORA_PATH = "{OUTPUT_DIR}"',
            "",
            "# Load tokenizer",
            "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)",
            "",
            "# Load base model",
            "base_model = AutoModelForCausalLM.from_pretrained(",
            "    MODEL_NAME,",
            "    trust_remote_code=True,",
            "    torch_dtype=torch.float32",
            ")",
            "",
            "# Load LoRA adapter",
            "model = PeftModel.from_pretrained(base_model, LORA_PATH)",
            "model.eval()",
            "```",
            "",
            "## Fine-tuning Details",
            f"- LoRA Rank (r): 8",
            f"- LoRA Alpha: 16",
            f"- Target Modules: q_proj, k_proj, v_proj, o_proj",
            f"- Learning Rate: {LEARNING_RATE}",
            f"- Batch Size: {BATCH_SIZE}",
            f"- Max Length: {MAX_LENGTH}",
            f"- Epochs: {EPOCHS}",
            "",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        readme_content = "\n".join(readme_lines)
        readme_file = OUTPUT_DIR / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        logger.info(f"📄 Created README: {readme_file}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        return
    
    # =====================================================
    # 8. FINAL MESSAGE
    # =====================================================
    print("\n" + "="*80)
    print("🎉 FINE-TUNING COMPLETED SUCCESSFULLY!")
    print("="*80)
    logger.info(f"📁 Model saved to: {OUTPUT_DIR}")
    logger.info(f"📊 Total trainable parameters: {trainable_params:,}")
    logger.info(f"⏱️  Training completed")
    
    print("\n📋 NEXT STEPS:")
    print(f"1. Your fine-tuned model is ready at: {OUTPUT_DIR}")
    print("2. Test it with chat_lora.py")
    print("3. Update chat_lora.py with this path:")
    print(f'   LORA_PATH = "{OUTPUT_DIR.absolute()}"')
    print("\n" + "="*80)

# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)