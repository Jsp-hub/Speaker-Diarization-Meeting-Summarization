import nltk
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from evaluate import load
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType

nltk.download('punkt')
torch.cuda.empty_cache()


dataset = load_dataset("samsum",trust_remote_code=True)
dataset['train'] = (
    dataset['train']
    .shuffle(seed=42)
    .select(range(2000))
)
metric = load("rouge")



def preprocess_meeting_transcript(example):
    
    if ":" not in example["dialogue"]:
        lines = example["dialogue"].split("\n")
        processed = []
        for i, line in enumerate(lines):
            if line.strip():
                processed.append(f"Speaker {i%3 + 1}: {line.strip()}")
        example["dialogue"] = "\n".join(processed)
    return example

dataset = dataset.map(preprocess_meeting_transcript)



model_checkpoint = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,                         
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, peft_config)



def preprocess_function(examples):
    inputs = [doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(
        inputs,
        max_length=1024,  
        truncation=True,
        padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"],
            max_length=180,  
            truncation=True,
            padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    
    decoded_preds = [pred.replace("Speaker 1:", "").replace("Speaker 2:", "").strip()
                    for pred in decoded_preds]
    decoded_labels = [label.replace("Speaker 1:", "").replace("Speaker 2:", "").strip()
                     for label in decoded_labels]

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}



batch_size = 8  
args = Seq2SeqTrainingArguments(
    output_dir="meeting-summarization-model",
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=2,  
    weight_decay=0.01,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True if torch.cuda.is_available() else False,
    report_to="none",
    warmup_steps=500,
    logging_steps=100,
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("final-meeting-summarization-model")


tokenizer.save_pretrained("meeting-summarization-tokenizer")