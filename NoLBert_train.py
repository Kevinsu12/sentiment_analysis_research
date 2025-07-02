import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

def main():
    df = pd.read_csv("data/Bert_training/processed_text.csv")        
    ds = Dataset.from_pandas(df, preserve_index=False)

    split = ds.train_test_split(test_size=0.05, seed=42)
    train_ds, val_ds = split["train"], split["test"]

    MODEL_NAME = "alikLab/NoLBERT"
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    model      = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    def tokenize_full(batch):
        return tokenizer(
            batch["text"],
            truncation =True,             
            padding="max_length",         
            max_length=512,
            return_special_tokens_mask=True,
        )

    train_tok = train_ds.map(
        tokenize_full,
        batched=True,
        remove_columns=["text"]
    )
    val_tok = val_ds.map(
        tokenize_full,
        batched=True,
        remove_columns=["text"]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="nolbert-finetuned",
        num_train_epochs=3.0,

        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,

        eval_strategy="steps",
        eval_steps=5000,

        logging_strategy="steps",
        logging_steps=1000,

        save_strategy="steps",
        save_steps=5000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("nolbert-earned-calls")


if __name__ == "__main__":
    main()