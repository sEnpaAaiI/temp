import torch
import os
import sys
import json
import IPython
from datetime import datetime
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer


# set device
device = 'cuda' if torch.cuda.is_available() else "cpu"
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch


from datasets import load_dataset

# Convert dataset to OAI messages
system_message = """You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.
SCHEMA:
{schema}"""

def create_conversation(sample):
  return {
    "messages": [
      {"role": "system", "content": system_message.format(schema=sample["context"])},
      {"role": "user", "content": sample["question"]},
      {"role": "assistant", "content": sample["answer"]}
    ]
  }

# Load dataset from the hub
dataset = load_dataset("b-mc2/sql-create-context", split="train")
dataset = dataset.shuffle().select(range(12500))

# Convert dataset to OAI messages
dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False)

# split dataset into 10,000 training samples and 2,500 test samples
dataset = dataset.train_test_split(test_size=2500/12500)

print(dataset["train"][345]["messages"])

# save datasets to disk
dataset["train"].to_json("train_dataset.json", orient="records")
dataset["test"].to_json("test_dataset.json", orient="records")


from datasets import load_dataset

# Load jsonl data from disk for sql
dataset = load_dataset("json", data_files="train_dataset.json", split="train")



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import setup_chat_format

# Hugging Face model id
model_id = "mistralai/Mistral-7B-Instruct-v0.1" #01 march 2024 AND 10/03/2024

#model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1" # 04/03/2024

#model_id = "meta-llama/Llama-2-7b-hf" #08 march 2024
# meta-llama/Llama-2-7b-hf

#model_id = "abacusai/Smaug-72B-v0.1" #04 march 2024

#model_id = "codellama/CodeLlama-7b-hf"

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
#     attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=True)
tokenizer.padding_side = 'right' # to prevent warnings

# We redefine the pad_token and pad_token_id with out of vocabulary token (unk_token)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id

# # set chat template to OAI chatML, remove if you start from a fine-tuned model
model, tokenizer = setup_chat_format(model, tokenizer)

from peft import LoraConfig

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)


#!pip install transformers==4.36.2 --quiet
#!pip install transformers accelerate --quiet

from transformers import TrainingArguments


# frankmorales2020/Mistral-7B-text-to-sql
args = TrainingArguments(
    # frankmorales2020/Mistral-7B-text-to-sql-flash-attention-2
    output_dir="Mistral-7B-text-to-sql-flash-attention-2-first-try",    # directory to save and repository id
    #output_dir="Mixtral-8x7B-Instruct-v0.1-text-to-sql", # 04/03/2024
    # Llama-2-7b-hf
    #output_dir="Llama-2-7b-hf-text-to-sql-flash-attention-2", # 08/03/2024
    #output_dir="Llama-2-7b-hf-squad2-flash-attention-2",

    #output_dir="Smaug-72B-v0.1-text-to-sql"
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=3,          # batch size per device during training
    gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
#     bf16=True,                              # use bfloat16 precision
#     tf32=True,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)


repo_id = "my-awesome-model-poc"


from trl import SFTTrainer
max_seq_length = 3072 # max sequence length for model and packing of the dataset


trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    #dataset_text_field="text",
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
        #"dataset_text_field": "text",  # Specify the text field in the dataset
    }
)