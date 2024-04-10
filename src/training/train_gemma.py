import os
import transformers
import torch
import numpy as np
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, GemmaTokenizer
from dotenv import load_dotenv
from preprocess_data import preprocess_file
from datasets import Dataset, load_dataset
from sklearn.model_selection import KFold

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    print(eval_pred)
    print(logits.shape)
    print(labels.shape)
    return {"loss": logits.loss.mean().item(), 'logits': logits.shape, 'labels': labels.shape}

def formatting_func(example):
    text = f"input: {example['input']}\ntarget: {example['target']}"
    return [text]

if __name__ == "__main__":
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv('TOKEN')
    model_id = "google/gemma-2b"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=os.environ['HF_TOKEN'])
    lora_config = LoraConfig(
        r = 8,
        target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type = "CAUSAL_LM",
    )
    tokenizer.padding_side = 'right'

    k = 5
    epoch = 100
    save_path = os.path.join("..", "models")
    os.makedirs(save_path, exist_ok=True)
    data_path = os.path.join("data", "stories3", "data.txt")
    stories = preprocess_file(data_path)
    output_dir = os.path.join('outputs')

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    stories_array = np.array(stories)

    # k-fold cross validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(stories_array)):
        print(f"Starting fold {fold + 1}/{k}")

        # Split the data for the current fold
        train_stories = stories_array[train_idx].tolist()
        val_stories = stories_array[val_idx].tolist()
        train_data = Dataset.from_list(train_stories)
        val_data = Dataset.from_list(val_stories)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"":0},
            token=os.environ['HF_TOKEN']
        )
        
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=transformers.TrainingArguments(
                do_train=True,
                do_eval=True,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                warmup_steps=2,
                max_steps=epoch,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=1,
                output_dir=output_dir,
                eval_steps=100,
                optim="paged_adamw_8bit"
            ),
            peft_config=lora_config,
            formatting_func=formatting_func,
            compute_metrics=compute_metrics,
            # packing=True,
        )
        trainer.train()
        trainer.evaluate()

        epoch_save_path = os.path.join(save_path, f'fold_{fold+1}_epoch_{epoch}_gemma.pth')
        trainer.save_model(epoch_save_path)
        print(f'Model saved to {epoch_save_path}')