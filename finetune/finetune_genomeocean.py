import torch
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling  # Added DataCollatorForLanguageModeling
from datasets import load_dataset
from Bio import SeqIO
import os

# Create a writable cache dir inside /data (if it doesn't exist)
cache_dir = "/data/hf_cache"
os.makedirs(cache_dir, exist_ok=True)

# Preprocess FASTA to plain text (one sequence per line)
input_fasta = "generated_sequences.fasta"
output_txt = "dna_sequences.txt"
sequences = [str(record.seq) for record in SeqIO.parse(input_fasta, "fasta")]
with open(output_txt, "w") as f:
    f.write("\n".join(sequences))
print(f"Preprocessed {len(sequences)} sequences to {output_txt}")

# Load model and tokenizer with custom cache_dir
model_name = "DOEJGI/GenomeOcean-100M"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="auto", torch_dtype=torch.bfloat16)

# Set pad_token if not already set (required for proper padding in CausalLM)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# Load preprocessed data
dataset = load_dataset("text", data_files=output_txt)

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset.set_format("torch")

# Data collator for CausalLM (shifts labels automatically)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training args (small for demo with 10 sequences)
training_args = TrainingArguments(
    output_dir="./finetuned_results",
    num_train_epochs=3,  # A few epochs since dataset is tiny
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    weight_decay=0.001,
    warmup_ratio=0.03,
    logging_steps=1,  # Log often for small data
    save_steps=5,
    evaluation_strategy="no",
    report_to="none",  # Disable W&B (or remove if you want logging)
)

# Trainer with data_collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,  # Added this
)

# Train
trainer.train()

# Save
model.save_pretrained("./finetuned_genomeocean")
tokenizer.save_pretrained("./finetuned_genomeocean")

print("Fine-tuning complete!")
