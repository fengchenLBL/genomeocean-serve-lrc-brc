# Fine-Tuning GenomeOcean with Apptainer

This README provides a simple guide to fine-tune the GenomeOcean model (e.g., __`DOEJGI/GenomeOcean-100M`__) using Apptainer (converted from the Docker image __`fengchenlbl/genomeocean:latest`__). It uses LoRA for efficient fine-tuning on a small DNA sequence dataset in FASTA format. This demo is designed for quick testing during meetings or workshops, with a tiny 10-sequence dataset—scale up for real applications.

## Prerequisites
- Apptainer installed on your system (common in HPC environments).
- NVIDIA GPU for faster training (use `--nv` flag for GPU passthrough).
- Docker image: `fengchenlbl/genomeocean:latest` (publicly available).
- A writable directory for data and caches (e.g., `/global/scratch/users/$USER/genomeocean-serve-lrc-brc/finetune` or a local `./tmp`).
- Sample dataset: Save the following FASTA content to `generated_sequences.fasta` in your data directory:

```
>seq1
ANNNNNNNNNGTATAGGGATAACCGCACATTGGACAGTAATATGTCCAATTTTCAAGATCAATAAATTGTTTGCACCGCGGGCATCGTTTCATTTTGTTGCCTCCTTTATAAAAATATCAAAATTGTAAGTATATAGGTTCTGTACTTGTTCTTTTTCTAATGTACCA
>seq2
ACCAGGGCCTTCAGGCCGATGCCCCGGCTGGCCATCAGCTTTGGCAGCATACTGGTTTCGGTCATGCCGGGGAGGGTGTTGATCTCCAGGAAGTAGGCGTTGTTCCCTTCGGTGATGATGAAATCCGACCGGGAGTAGCCGTCCAGGCCCAGCAGGTGGTAGACCTCCAGAGCAAGGGTCTGGATCTGTTTGGTCAGGTCATCCGGCAGCGGGGCAGGGCAGATCTCCTGGGTGCCGCTG
>seq3
ACCAGGGCGGCAAAGTCCTCGTTGGTGAGGCTCTTGGGGTCGGTCTCGTTGCTGATCTTGCAGTCCCGGACGATGGCCTTGGCGGTGGCAGGGGTGAGCCCCAGGGAAGCCAGGTTGGTCACCAGGCCCAGCCGCACCACAAAGTCCATCAGGGCGGAGACGCCGGTCTCCCCCAGGCCGGCAGCGGCGGTGAGGGCGGCCATCCGCTCCTCCCGGGAGGCGCTGGGGGCGTCGATGGCGATGGCG
>seq4
ACCAGGGCGCTGACCCTTTTCATAGCCAGCTCCAGGTTTTCAAGGGAGGAGGCGTAGGACAGGCGGATAAAGCCCTCGCCGCACTCGCCGAAGGCGCTGCCCGGCACTACCGCCACCTTTTCCTCCCGGATGAGCCGGCGGCAGAAATCCTCGCTGCTCATGCCGGTTTCTTTAATGCAAGGAAAGCAGTAAAAGCTGCCCCTGGGCACCGGCACATCCAGCCCCATGCGGTTTAAGCCCTGCACCG
>seq5
ACCAGGGTGAGCACCAGCTCCCAGTTATTTTCCTCGGTTTTCCGAAAGAGGGAAAGCTGTGGGTACCAGGGGCTGTCCGGGCGGTCCAAAAGCCAACGCCAGGAGGCGCAGTAGGGCAGGAGCACCCACACCGGCTTGCCCAGAGCCCCGGCCAGGTGGGCCACCGAGGTATCCACCGTGATCACCAAATCCATGCAGGACAAAAGGCCCGCCGTATGGGAAAAATCTTCCAGG
>seq6
ACCAGGGCCAGTCCCTCCCTTATCTCCCCTTCTGTAGGAAGGGAGAAGTTCAGCCGGATACAGTTCTCATTCCCATATCCCAGGAACGGTCCTGGCATGACGGCTACTCCGCCCTTCAGCGCCAGCTCCAGAAGTCCCAGCGTATCCAGATTCCCCCGGGCCTTCAGCCAGAGGAACAATCCGCCTTCGGGGTGGTAGAACTGAAACTTTCCCTCCAGCTTTTCCTGG
>seq7
ACCAGGGCTTCCACGGCCTTTTCCAGCTGCGCCGGATCTTTGCCGCCCGCCGTGGCGAAATCCGGTTTTCCGCCGCCGTTTCCGCCGGCGATCTTGGCGATTTCACGAACAAGTTTTCCCGCATGGGCGCCCTTTTTGACCGCCTCCGGACCCACCATCACCAGCATATTGGCACGGCCCTCGCTGGCCGCGCCCAGAACAATGGTCTTGGGGCTGTCCACCTGTTTTCTCAG
>seq8
ANNNNNNNNNGAGCCAAACGCTCATAATCCACCAGATCAAAGGCAAACTTGATCGCATCACCATGAGCTCCACAGCCAAAACAGTGGTAAAACTGCTTTTCGGCATTTACGGTAAACGAGGGCGTTTTCTCATTGTGAAAGGGACAAAGCCCCAAAAAAGAACGTCCGCTGCGCTTGAGTTTT
>seq9
ACCAGGGCCTTGTAGCCGTCGATATTATTGGACTTGGGCGTGACGTGGCAGGTGTAGGTCTTTCCCTCCACGGACTCGTAATGGGCAAAGACCTCCCGGATCTTCCGCTCTATGTGGGCCAGCTCCTCGGGGGTGTTGTTGTGGGTGTAGTTGGTGATGTAGACCACGCCGTCCTTCATGGCGTAGGGGGCCAGACGGTTCATCCACTCCAGCAGCTTCA
>seq10
ACCAGGGCATAAATCCGCTGGTCTTCATCGTAATTGGAAAAGACGGAAAAGCTGATTTTCCCCTGGGTGGAGATGACGAAGACGGTGGGCACGGTAAAGGAGCTGTAGGTGCCGTAGGACACCATGAGGTCGCCGTAGCCGGGGGCGCTGATGGGATAGGACACGCCGTAGTCCTTGAGGAAGGCCTCGGCCTCCTCCTGGGAGCCGTCCCGGCGGTTGAGGCCCAGG
```

## Step 1: Convert Docker Image to Apptainer

#### Set up Apptainer cache directory:
```
mkdir -p /global/scratch/users/$USER/.apptainer
export APPTAINER_CACHEDIR=/global/scratch/users/$USER/.apptainer
```

#### Create a temporary directory for model cache:
```
git clone https://github.com/fengchenLBL/genomeocean-serve-lrc-brc.git
cd genomeocean-serve-lrc-brc/finetune
mkdir -p ./tmp
```


#### Pull and convert the Docker image to an Apptainer `.sif` file:

```bash
apptainer pull genomeocean.sif docker://fengchenlbl/genomeocean:latest
```

This creates a portable `genomeocean.sif` file.

## Step 2: Prepare Your Data and Script
- Place `generated_sequences.fasta` in your data directory (e.g., `./`).
- Create `finetune_genomeocean.py` in the same directory with the following content (this script preprocesses FASTA, loads the model, applies LoRA, tokenizes, trains, and saves):

```python
import torch
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
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
    data_collator=data_collator,
)

# Train
trainer.train()

# Save
model.save_pretrained("./finetuned_genomeocean")
tokenizer.save_pretrained("./finetuned_genomeocean")

print("Fine-tuning complete!")
```

**Notes on the Script:**
- Uses `GenomeOcean-100M` for quick demos (swap to `GenomeOcean-4B` for larger models, but requires more VRAM).
- Preprocesses FASTA using BioPython (assumed installed in the image).
- Downloads models/tokenizers to a writable cache (`/data/hf_cache`).
- Trains with LoRA (efficient adapter-based fine-tuning) for causal language modeling on DNA sequences.
- With 10 sequences, it runs in seconds; expect overfitting—use larger datasets for production.

## Step 3: Run the Fine-Tuning
Enter an interactive Apptainer shell with bindings for writable caches and data:

```bash
apptainer shell --nv --bind ./tmp:/workspace/model_cache -B ./:/data genomeocean.sif
```

Inside the shell (prompt: `Apptainer>`):
- If PEFT isn't installed: `pip install peft` (session-specific; add to script if needed for automation).
- Run the script: `python finetune_genomeocean.py`

For non-interactive execution (after testing):
```bash
apptainer exec --nv --bind ./tmp:/workspace/model_cache -B ./:/data genomeocean.sif python /data/finetune_genomeocean.py
```

**Bindings Explained:**
- `--bind ./tmp:/workspace/model_cache`: Makes the internal cache writable (avoids read-only errors).
- `-B /your/host/data/dir:/data`: Mounts your data directory (adjust path as needed).

## Example Output
Here's an example of a successful run:

```
Apptainer> python finetune_genomeocean.py
2025-10-29 09:07:44.716643: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-10-29 09:07:44.745952: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2025-10-29 09:07:44.745978: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2025-10-29 09:07:44.746836: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-10-29 09:07:44.751716: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 AVX_VNNI AMX_TILE AMX_INT8 AMX_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-10-29 09:07:45.773573: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Preprocessed 10 sequences to dna_sequences.txt
tokenizer_config.json: 1.14kB [00:00, 9.27MB/s]
tokenizer.json: 166kB [00:00, 337MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 695/695 [00:00<00:00, 10.9MB/s]
config.json: 1.13kB [00:00, 12.2MB/s]
model.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 239M/239M [00:02<00:00, 116MB/s]
generation_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 132/132 [00:00<00:00, 2.11MB/s]
Generating train split: 10 examples [00:00, 844.30 examples/s]
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 331.94 examples/s]
/opt/conda/envs/GO/lib/python3.11/site-packages/transformers/training_args.py:1525: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
  warnings.warn(
Detected kernel version 4.18.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
{'loss': 5.2004, 'grad_norm': 5.173176288604736, 'learning_rate': 0.0002, 'epoch': 0.8}                                                                        
{'loss': 4.9181, 'grad_norm': 6.224944114685059, 'learning_rate': 0.0001, 'epoch': 1.6}                                                                        
{'loss': 4.9742, 'grad_norm': 4.977142333984375, 'learning_rate': 0.0, 'epoch': 2.4}                                                                           
{'train_runtime': 3.3215, 'train_samples_per_second': 9.032, 'train_steps_per_second': 0.903, 'train_loss': 5.0309146245320635, 'epoch': 2.4}                  
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:03<00:00,  1.11s/it]
Fine-tuning complete!
```

**Notes:**
- TensorFlow warnings are benign (from the environment; ignore unless issues arise).
- Kernel warning: Upgrade host kernel to 5.5.0+ if training hangs in larger runs.
- Outputs: Check `./finetuned_results` for logs/checkpoints and `./finetuned_genomeocean` for the saved model/adapters.

## Troubleshooting
- **Read-Only Errors:** Ensure bindings provide writable paths for caches.
- **PEFT Missing:** Install via `pip install peft` inside the shell.
- **Resources:** ~4-8GB VRAM for `100M` model; use `load_in_8bit=True` in model loading for low-memory setups.
- **Merging Adapters:** Post-training, add code to merge LoRA into the base model for inference.

For questions, refer to the GenomeOcean repository or Hugging Face docs on transformers/PEFT.
