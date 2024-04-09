import os
import transformers
import torch
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, GemmaTokenizer
from dotenv import load_dotenv
from preprocess_data import preprocess_file
from datasets import Dataset, load_dataset

torch.cuda.empty_cache()

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv('TOKEN')

prompt = "Caroline. I learned her name through the phonebook, my shaking fingers carefully caressing its pages as I searched for the address I'd seen her at so many times."
model_path = os.path.join('..', 'models', 'fold_1_epoch_100_gemma.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ----
# LOAD LLM
model_id = "google/gemma-2b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     quantization_config=bnb_config,
#     device_map={"":0},
#     token=os.environ['HF_TOKEN']
# )

model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer.padding_side = 'right'

# load_model = PeftModel.from_pretrained(model, model_path)
# load_model = load_model.merge_and_unload()

inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=200)
nice_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(nice_output)

