"""
FINAL Fine-Tuning Script for Paper
Optimized for 418 samples with DeepSeek-Coder-1.3B
"""

import torch
from pathlib import Path
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

@dataclass
class ModelArguments:
    model_name: str = field(
        default="deepseek-ai/deepseek-coder-1.3b-base",
        metadata={"help": "1.3B model better for small datasets than 7B"}
    )
    use_4bit: bool = field(default=True)
    use_lora: bool = field(default=True)
    lora_r: int = field(default=64, metadata={"help": "Higher rank for complex code"})
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    merge_and_save: bool = field(default=True)

@dataclass
class DataArguments:
    dataset_path: str = field(default="derivative_dataset_537/finetuning")
    max_length: int = field(default=2048)

class FinalTrainer:
    def __init__(self, model_args, data_args, training_args):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = None
        self.model = None
        self.dataset = None
        Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self):
        print("\n📊 Loading dataset...")
        dataset_dir = Path(self.data_args.dataset_path)
        
        data_files = {
            'train': str(dataset_dir / "hf_train.jsonl"),
            'validation': str(dataset_dir / "hf_validation.jsonl"),
        }
        
        dataset = load_dataset('json', data_files=data_files)
        
        print(f"✓ Loaded {len(dataset['train'])} training samples")
        print(f"✓ Loaded {len(dataset['validation'])} validation samples")
        
        # Print sample
        sample = dataset['train'][0]
        print(f"\n📝 Sample:")
        print(f"  Input: {sample['input']}")
        print(f"  Output length: {len(sample['output'])} chars")
        
        self.dataset = dataset
        return dataset
    
    def load_model_and_tokenizer(self):
        print(f"\n🤖 Loading model: {self.model_args.model_name}")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"✓ Tokenizer loaded (vocab: {len(self.tokenizer)})")
        
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
                print("✓ 4-bit quantization enabled")
            except ImportError:
                print("⚠️ bitsandbytes not available, using fp16")
                self.model_args.use_4bit = False
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        if self.model_args.use_4bit:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.training_args.gradient_checkpointing
            )
        
        if self.training_args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        if self.model_args.use_lora:
            print(f"✓ Applying LoRA (r={self.model_args.lora_r}, α={self.model_args.lora_alpha})")
            
            # Comprehensive target modules for DeepSeek
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
            
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
    
    def tokenize_dataset(self):
        print("\n🔤 Tokenizing dataset...")

        def format_and_tokenize(examples):
            """Format with instruction-following template for DeepSeek"""
            prompts = []
            responses = []

            for i in range(len(examples['instruction'])):
                # Use a clear instruction format that matches inference
                # DeepSeek format: instruction -> response
                instruction = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Generate Manim animation code for the derivative of: f(x) = {examples['input'][i]}

### Response:
"""
                response = f"{examples['output'][i]}{self.tokenizer.eos_token}"

                # Combine for full sequence
                full_prompt = instruction + response
                prompts.append(full_prompt)
                responses.append(response)

            # Tokenize full prompts
            result = self.tokenizer(
                prompts,
                truncation=True,
                max_length=self.data_args.max_length,
                padding="max_length",
                return_tensors=None,
            )

            # For causal LM, labels = input_ids
            # The model learns to predict next token given previous tokens
            result["labels"] = result["input_ids"].copy()

            return result
        
        tokenized_dataset = self.dataset.map(
            format_and_tokenize,
            batched=True,
            num_proc=None,
            remove_columns=self.dataset["train"].column_names,
            desc="Tokenizing"
        )
        
        print(f"✓ Tokenized {len(tokenized_dataset['train'])} training samples")
        return tokenized_dataset
    
    def train(self):
        print("\n" + "="*70)
        print("FINAL FINE-TUNING FOR PAPER")
        print("="*70)
        
        self.load_dataset()
        self.load_model_and_tokenizer()
        tokenized_dataset = self.tokenize_dataset()
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            callbacks = [EarlyStoppingCallback(early_stopping_patience = 2)]
        )
        
        print("\n🚀 Training started...\n")
        train_result = trainer.train()
        
        print("\n💾 Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.training_args.output_dir)
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        if self.model_args.use_lora and self.model_args.merge_and_save:
            print("\n🔄 Merging LoRA weights...")
            merged_dir = Path(self.training_args.output_dir) / "merged_model"
            merged_dir.mkdir(exist_ok=True)
            model = trainer.model.merge_and_unload()
            model.save_pretrained(merged_dir)
            self.tokenizer.save_pretrained(merged_dir)
            print(f"✓ Merged model saved to: {merged_dir}")
        
        print("\n✅ Training complete!")
        print(f"\nFinal metrics:")
        print(f"  Train loss: {metrics['train_loss']:.4f}")
        print(f"  Epochs: {metrics['epoch']}")
        
        return trainer

def main():
    print("\n" + "="*70)
    print("DERIVATIVE ANIMATOR FINE-TUNING")
    print("Optimized for 418 samples")
    print("="*70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\n⚠️ No GPU detected!")
        return
    
    # Model configuration - OPTIMIZED
    model_args = ModelArguments(
        model_name="deepseek-ai/deepseek-coder-1.3b-base",  # Smaller = better for 418 samples
        use_4bit=True,
        use_lora=True,
        lora_r=64,  # High rank for complex code generation
        lora_alpha=128,
        lora_dropout=0.05,
    )
    
    data_args = DataArguments(
        dataset_path="derivative_dataset_537/finetuning",
        max_length=2048,
    )
    
    # Training configuration - OPTIMIZED for small dataset
    training_args = TrainingArguments(
        output_dir="./derivative-animator-deepseek-1.3b",
        
        # Training duration
        num_train_epochs=5,  # More epochs for small dataset
        
        # Batch sizes - smaller for better learning
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Effective batch = 1*16 = 16
        
        # Learning rate - higher for smaller model
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        
        # Evaluation
        logging_steps=5,
        save_steps=25,
        save_strategy="steps",
        eval_steps=25,
        eval_strategy="steps",
        
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Optimization
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        max_grad_norm=1.0,
        
        report_to="none",
    )
    
    # Verify dataset exists
    dataset_path = Path(data_args.dataset_path)
    if not dataset_path.exists():
        print(f"\n❌ Dataset not found: {dataset_path}")
        print("Please run finetuning_data_preparation.py first")
        return
    
    required_files = ["hf_train.jsonl", "hf_validation.jsonl"]
    missing = [f for f in required_files if not (dataset_path / f).exists()]
    if missing:
        print(f"\n❌ Missing files: {missing}")
        return
    
    # Train
    try:
        trainer_obj = FinalTrainer(model_args, data_args, training_args)
        trainer_obj.train()
        
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. Test the model:")
        print("   python test_finetuned_model.py")
        print("\n2. Run comparison:")
        print("   python compare_models_final.py")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()