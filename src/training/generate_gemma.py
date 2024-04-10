import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

class GemmaStoryModel():
    def __init__(self, model_path):
        torch.cuda.empty_cache()
        load_dotenv()
        os.environ["HF_TOKEN"] = os.getenv('TOKEN')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.model_id = "google/gemma-2b"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=os.environ['HF_TOKEN'])
        self.tokenizer.padding_side = 'right'
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)

    def tokenize_text_for_gemma(self, text):
        return self.tokenizer(text, return_tensors="pt").to(self.device)
    
    def generate_story(self, input, max_response_length=200):
        print('generating...')
        text = self.tokenize_text_for_gemma(input)
        response = self.model.generate(**text, max_new_tokens=max_response_length)
        print('generated!')
        return self.tokenizer.decode(response[0], skip_special_tokens=True)

def main():
    prompt = "Erwin could feel Neitsh's anger from the next tent over. The cleric had disappeared into it after barely eating anything, claiming to be tired."
    model_path = os.path.join('models', 'fold_2_epoch_100_gemma.pth')
    generator = GemmaStoryModel(model_path)
    print(generator.generate_story(prompt))

if __name__=='__main__':
    main()