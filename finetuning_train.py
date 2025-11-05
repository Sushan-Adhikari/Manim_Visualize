"""
Fixed Fine-Tuning Script for DerivativeAnimator
Addresses: bitsandbytes, dataset loading, and HuggingFace format issues
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate

# Optional: WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  WandB not available. Install with: pip install wandb")

@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name: str = field(
        default="codellama/CodeLlama-7b-hf",
        metadata={"help": "Base model to fine-tune"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Use 4-bit quantization (saves VRAM)"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Use LoRA for parameter-efficient fine-tuning"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    merge_and_save: bool = field(
        default=True,
        metadata={"help": "Merge LoRA weights after training"}
    )

@dataclass
class DataArguments:
    """Arguments for dataset configuration"""
    dataset_path: str = field(
        default="derivative_dataset_537/finetuning",
        metadata={"help": "Path to finetuning directory (containing JSONL files)"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "Number of workers for preprocessing"}
    )

class DerivativeAnimatorTrainer:
    """Main training class with bug fixes"""
    
    def __init__(self, 
                 model_args: ModelArguments,
                 data_args: DataArguments,
                 training_args: TrainingArguments):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
        # Create output directory
        Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)
        
    def setup_wandb(self):
        """Initialize Weights & Biases logging"""
        if not WANDB_AVAILABLE or self.training_args.report_to != "wandb":
            return
        
        wandb.init(
            project="derivative-animator",
            name=f"finetune-{self.model_args.model_name.split('/')[-1]}",
            config={
                "model": self.model_args.model_name,
                "epochs": self.training_args.num_train_epochs,
                "batch_size": self.training_args.per_device_train_batch_size,
                "learning_rate": self.training_args.learning_rate,
            }
        )
    
    def load_dataset(self):
        """Load dataset - FIXED to use correct JSONL files"""
        print("\n📊 Loading dataset...")
        
        dataset_dir = Path(self.data_args.dataset_path)
        
        # Check for JSONL files (correct format)
        train_file = dataset_dir / "hf_train.jsonl"
        val_file = dataset_dir / "hf_validation.jsonl"
        test_file = dataset_dir / "hf_test.jsonl"
        
        if not train_file.exists():
            raise FileNotFoundError(
                f"Training file not found: {train_file}\n"
                f"Please ensure finetuning_data_preparation.py was run successfully."
            )
        
        # Load dataset using JSONL files
        data_files = {
            'train': str(train_file),
            'validation': str(val_file),
        }
        
        if test_file.exists():
            data_files['test'] = str(test_file)
        
        dataset = load_dataset('json', data_files=data_files)
        
        print(f"✓ Loaded {len(dataset['train'])} training samples")
        print(f"✓ Loaded {len(dataset['validation'])} validation samples")
        if 'test' in dataset:
            print(f"✓ Loaded {len(dataset['test'])} test samples")
        
        # Print sample to verify format
        print("\n📝 Sample from dataset:")
        sample = dataset['train'][0]
        print(f"  Instruction length: {len(sample.get('instruction', ''))}")
        print(f"  Input: {sample.get('input', '')[:50]}...")
        print(f"  Output length: {len(sample.get('output', ''))}")
        
        self.dataset = dataset
        return dataset
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer - FIXED bitsandbytes handling"""
        print(f"\n🤖 Loading model: {self.model_args.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"✓ Tokenizer loaded (vocab size: {len(self.tokenizer)})")
        
        # Quantization config - with fallback
        quantization_config = None
        if self.model_args.use_4bit:
            try:
                import bitsandbytes
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                print("✓ Using 4-bit quantization (NF4)")
            except ImportError:
                print("⚠️  bitsandbytes not available, loading in fp16 instead")
                self.model_args.use_4bit = False
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Prepare for training
        if self.model_args.use_4bit:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.training_args.gradient_checkpointing
            )
        
        # Enable gradient checkpointing
        if self.training_args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Apply LoRA
        if self.model_args.use_lora:
            print(f"✓ Applying LoRA (r={self.model_args.lora_r})")
            
            target_modules = self._get_target_modules()
            
            lora_config = LoraConfig(
                r=self.model_args.lora_r,
                lora_alpha=self.model_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.model_args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        return self.model, self.tokenizer
    
    def _get_target_modules(self):
        """Get target modules for LoRA"""
        model_name_lower = self.model_args.model_name.lower()
        
        if "llama" in model_name_lower or "codellama" in model_name_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "mistral" in model_name_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "deepseek" in model_name_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    def tokenize_dataset(self):
        """Tokenize dataset - FIXED prompt format"""
        print("\n🔤 Tokenizing dataset...")
        
        def format_and_tokenize(examples):
            """Format with proper instruction template"""
            # Build prompts
            prompts = []
            for i in range(len(examples['instruction'])):
                prompt = f"""{examples['instruction'][i]}

### Input:
{examples['input'][i]}

### Output:
{examples['output'][i]}"""
                prompts.append(prompt)
            
            # Tokenize
            result = self.tokenizer(
                prompts,
                truncation=True,
                max_length=self.data_args.max_length,
                padding="max_length",
                return_tensors=None,
            )
            
            # Set labels
            result["labels"] = result["input_ids"].copy()
            return result
        
        tokenized_dataset = self.dataset.map(
            format_and_tokenize,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=self.dataset["train"].column_names,
            desc="Tokenizing"
        )
        
        print(f"✓ Tokenized {len(tokenized_dataset['train'])} training samples")
        
        return tokenized_dataset
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("STARTING FINE-TUNING")
        print("="*70)
        
        # Setup
        if self.training_args.report_to == "wandb":
            self.setup_wandb()
        
        self.load_dataset()
        self.load_model_and_tokenizer()
        tokenized_dataset = self.tokenize_dataset()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Callbacks
        callbacks = []
        if self.training_args.load_best_model_at_end:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.001
                )
            )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            callbacks=callbacks,
        )
        
        # Train
        print("\n🚀 Training started...\n")
        train_result = trainer.train()
        
        # Save
        print("\n💾 Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.training_args.output_dir)
        
        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Merge if using LoRA
        if self.model_args.use_lora and self.model_args.merge_and_save:
            self._merge_and_save(trainer)
        
        print("\n✅ Training complete!")
        return trainer
    
    def _merge_and_save(self, trainer):
        """Merge LoRA weights"""
        print("\n🔄 Merging LoRA weights...")
        merged_dir = Path(self.training_args.output_dir) / "merged_model"
        merged_dir.mkdir(exist_ok=True)
        
        model = trainer.model.merge_and_unload()
        model.save_pretrained(merged_dir)
        self.tokenizer.save_pretrained(merged_dir)
        
        print(f"✓ Merged model saved to: {merged_dir}")

def main():
    """Main function"""
    print("\n" + "="*70)
    print("DERIVATIVEANIMATOR FINE-TUNING (FIXED)")
    print("="*70)
    
    # Check bitsandbytes
    try:
        import bitsandbytes
        print("✓ bitsandbytes available")
    except ImportError:
        print("⚠️  bitsandbytes not installed. Will use fp16 instead of 4-bit.")
        print("   Install with: pip install bitsandbytes")
    
    # Configuration
    model_args = ModelArguments(
        model_name="codellama/CodeLlama-7b-hf",
        use_4bit=True,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
    )
    
    data_args = DataArguments(
        dataset_path="derivative_dataset_537/finetuning",
        max_length=2048,
        preprocessing_num_workers=4
    )
    
    training_args = TrainingArguments(
        output_dir="./derivative-animator-codellama-7b",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        
        logging_steps=10,
        save_steps=100,
        save_strategy="steps",
        eval_steps=100,
        evaluation_strategy="steps",
        
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        
        report_to="none",
    )
    
    # Verify dataset
    dataset_path = Path(data_args.dataset_path)
    if not dataset_path.exists():
        print(f"\n❌ Dataset directory not found: {dataset_path}")
        return
    
    required_files = ["hf_train.jsonl", "hf_validation.jsonl"]
    missing_files = [f for f in required_files if not (dataset_path / f).exists()]
    if missing_files:
        print(f"\n❌ Missing required files: {missing_files}")
        print("Run finetuning_data_preparation.py first")
        return
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  No GPU detected")
    
    # Train
    try:
        trainer_obj = DerivativeAnimatorTrainer(model_args, data_args, training_args)
        trainer_obj.train()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()