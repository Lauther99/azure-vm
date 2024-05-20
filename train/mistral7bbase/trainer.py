# pip install torch transformers huggingface-hub accelerate trl pandas datasets peft bitsandbytes openpyxl
import os
import pandas as pd
import datasets as ds
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer
from huggingface_hub import login

login(token="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")
# Run with accelerate launch mistral7b-ft.py

mistral_instruct = "mistralai/Mistral-7B-Instruct-v0.2"
llama3_base = "meta-llama/Meta-Llama-3-8B"
llama3_instruct="meta-llama/Meta-Llama-3-8B-Instruct"

model_name = llama3_instruct

training_strategy = "DDP"  # "FSDP" o "DDP"
is_quantized = True
torch_type = torch.bfloat16

if training_strategy == "FSDP":
    output_dir = "./ft-models/" + model_name.split("/")[-1] + "-adapters"
else:
    output_dir = ("./ft-models/" + model_name.split("/")[-1] + "-" + str(torch_type).split("torch.")[-1] if is_quantized else "fp32")
    


# Cargamos el dataset
def get_dataset(model_name):
    row_name = model_name.split("/")[-1] + "-texts"
    current_dir = os.path.dirname(__file__)
    
    excel_path = os.path.join(current_dir, "./data/data.xlsx")
    df = pd.read_excel(excel_path, sheet_name="train")
    
    df_shuffled = df.sample(frac=1, random_state=42)
    
    texts = []
    
    for _, row in df_shuffled.iterrows():
        texts.append(row[row_name])

    texts_df = pd.DataFrame(texts, columns=["texts"])
    return ds.Dataset.from_pandas(texts_df)


# Cargamos el modelo
def get_model_and_tokenizer(model_name):
    if is_quantized:
        print("Cargando el modelo en: " + str(torch_type).split("torch.")[-1])
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir="",
            use_cache=False,
            torch_dtype=torch_type,
            low_cpu_mem_usage=True,
        )
    else:
        print("Cargando el modelo en: fp32")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir="",
            use_cache=False,
            low_cpu_mem_usage=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return (model, tokenizer)


# SFT Configuracion.
def train(model, tokenizer, dataset, output_dir):
    peft_config = LoraConfig(
        r=8,  # Set the rank of the LoRA projection matrix.
        lora_alpha=16,  # Set the alpha parameter for the LoRA projection matrix.
        lora_dropout=0.05,  # Set the dropout rate for the LoRA projection matrix.
        bias="none",  # Set the bias term to "none".
        task_type="CAUSAL_LM",  # Set the task type to "CAUSAL_LM".
    )

    training_args = TrainingArguments(
        output_dir=output_dir,  # Set the output directory for the training run.
        per_device_train_batch_size=64,  # Set the per-device training batch size.
        # gradient_accumulation_steps=2,  # Set the number of gradient accumulation steps.
        optim="paged_adamw_32bit",  # Set the optimizer to use.
        learning_rate=35e-5,  # Set the learning rate.
        lr_scheduler_type="cosine",  # Set the learning rate scheduler type.
        logging_steps=10,  # Set the logging steps.
        num_train_epochs=100, # Set the number of training epochs.
        # max_steps=381,  # Set the maximum number of training steps.
        fp16=not torch.cuda.is_bf16_supported(),  # Enable fp16 training.
        bf16=torch.cuda.is_bf16_supported(),
        warmup_ratio=0.03,  # Proporción de pasos para un calentamiento lineal (de 0 a tasa de aprendizaje)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        # group_by_length=True,  # Ahorra memoria y acelera considerablemente el entrenamiento
        save_strategy="steps",  # Set the save strategy.
        save_steps=25,  # Guardar punto de control cada X pasos de actualización
    )

    trainer = SFTTrainer(
        model=model,  # Set the model to be trained.
        tokenizer=tokenizer,  # Set the tokenizer.
        train_dataset=dataset,  # Set the training dataset.
        peft_config=peft_config,  # Set the PEFT configuration.
        dataset_text_field="texts",  # Set the name of the text field in the dataset.
        max_seq_length=2609,  # Cuando es None, el max_seq_len vendrá determinado por la secuencia más larga de un lote
        args=training_args,  # Set the training arguments.
        packing=False,  # Disable packing.
        # max_seq_length=1024 # Set the maximum sequence length.
    )

    trainer.model.print_trainable_parameters()

    trainer.train()

    print("=" * 10 + "=>" + "Guardando el modelo")
    trainer.save_model(output_dir)
    print("=" * 10 + "=>" + "Guardando el tokenizer")
    tokenizer.save_pretrained(f"{output_dir}/tokenizer")


dataset = get_dataset(model_name)
model, tokenizer = get_model_and_tokenizer(model_name)
train(model, tokenizer, dataset, output_dir)
