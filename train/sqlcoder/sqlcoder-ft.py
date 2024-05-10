import pandas as pd
import os
import datasets as ds

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer

# Run with accelerate launch sqlcoder-ft.py

# from accelerate import PartialState
# device_string = PartialState().process_index


# Ruta al archivo Excel en Google Drive
filename = "./data.xlsx"

# Search for the file
texts = []

df = pd.read_excel(filename, sheet_name="texts", usecols=["texts_train", "outputs_train"]).dropna()
for _, row in df.iterrows():
    # Agregar la fila completa o datos específicos de la fila a la lista
    q = row["texts_train"] + "\n" + row["outputs_train"] + "\n" + "[/SQL]"
    texts.append(q)
texts_df = pd.DataFrame(texts, columns=["texts"])
formatted_data = ds.Dataset.from_pandas(texts_df)

model_name = "defog/sqlcoder-7b-2"

# Importamos el tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Set the padding token to be the same as the end of sentence token.
tokenizer.pad_token = tokenizer.eos_token

# Definimos los paramétros para bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)

# Importamos el modelo pre-entrenado
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir='',
    use_cache = False,
    # attn_implementation="flash_attention_2",
    # torch_dtype = getattr(torch, "float16"),
    quantization_config=bnb_config,
    # device_map={'':device_string},
    # device_map="auto",
    low_cpu_mem_usage=True,  # Reduccion del consumo de cpu y memoria al leer el modelo
)

# Set the temperature for pretraining to 1.
model.config.pretraining_tp = 1

# Define the PEFT configuration.
peft_config = LoraConfig(
    r=8, # Set the rank of the LoRA projection matrix.
    lora_alpha=16, # Set the alpha parameter for the LoRA projection matrix.
    lora_dropout=0.05, # Set the dropout rate for the LoRA projection matrix.
    bias="none", # Set the bias term to "none".
    task_type="CAUSAL_LM" # Set the task type to "CAUSAL_LM".
)


output_dir="./ft-models/sql-coder-7b-2-peft"

training_args = TrainingArguments(
    output_dir=output_dir, # Set the output directory for the training run.

    per_device_train_batch_size=8, # Set the per-device training batch size.
    gradient_accumulation_steps=2, # Set the number of gradient accumulation steps.

    optim="paged_adamw_32bit", # Set the optimizer to use.
    learning_rate=2e-4, # Set the learning rate.
    lr_scheduler_type = "cosine", # Set the learning rate scheduler type.
    save_strategy="epoch", # Set the save strategy.
    logging_steps=25, # Set the logging steps.
    num_train_epochs=5, # Set the number of training epochs.
    max_steps=10, # Set the maximum number of training steps.
    fp16=True, # Enable fp16 training.
    bf16 = False,

    warmup_ratio = 0.03, # Proporción de pasos para un calentamiento lineal (de 0 a tasa de aprendizaje)

    gradient_checkpointing = True,
    gradient_checkpointing_kwargs = {"use_reentrant": True},
    
    group_by_length = True, # Ahorra memoria y acelera considerablemente el entrenamiento
    save_steps = 0, # Guardar punto de control cada X pasos de actualización
    )

# Initialize the SFTTrainer.
trainer = SFTTrainer(
    model=model, # Set the model to be trained.
    train_dataset=formatted_data, # Set the training dataset.
    peft_config=peft_config, # Set the PEFT configuration.
    dataset_text_field="texts", # Set the name of the text field in the dataset.
    max_seq_length=None, # Cuando es None, el max_seq_len vendrá determinado por la secuencia más larga de un lote
    args=training_args, # Set the training arguments.
    tokenizer=tokenizer, # Set the tokenizer.
    packing=False, # Disable packing.
    # max_seq_length=1024 # Set the maximum sequence length.
)

trainer.model.print_trainable_parameters()
if getattr(trainer.accelerator.state, "fsdp_plugin", None):
    from peft.utils.other import fsdp_auto_wrap_policy

    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)


# Iniciamos el entrenamiento
trainer.train()

print("="*10 + "Guardando el modelo" + "="*10)
trainer.model.save_pretrained(output_dir)
print("="*10 + "Guardando el tokenizer" + "="*10)
tokenizer.save_pretrained(f"{output_dir}/tokenizer")
