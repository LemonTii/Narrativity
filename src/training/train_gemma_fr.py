import os
import transformers
import torch
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, GemmaTokenizer
from dotenv import load_dotenv
from preprocess_data import preprocess_file


from datasets import Dataset, load_dataset


load_dotenv()
os.environ["HF_TOKEN"] = os.getenv('TOKEN')

# ----
# LOAD LLM
model_id = "google/gemma-2b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"":0}, # TODO: josh peepoo sok was here :P
    token=os.environ['HF_TOKEN']
)

# TEST MODEL WORKS
def test_model_works():
    prompt = "Caroline. I learned her name through the phonebook, my shaking fingers carefully caressing its pages as I searched for the address I'd seen her at so many times."
    device = "cuda"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print(f"DEBUG inputs={inputs}")

    outputs = model.generate(**inputs, max_new_tokens=20)
    print(f"DEBUG outputs={outputs}")

    nice_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"DEBUG nice_output={nice_output}")

# test_model_works()


# TRAIN MODEL
lora_config = LoraConfig(
    r = 8,
    target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type = "CAUSAL_LM",
)

data_path = os.path.join("data", "stories3", "data.txt")
stories = preprocess_file(data_path)
train_data = Dataset.from_list(stories)


def formatting_func(example):
    text = f"input: {example['input']}\ntarget: {example['target']}"
    return [text]


trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
)

trainer.train()

# TODO: save model

test_model_works()