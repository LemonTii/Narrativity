import os
import re
import nltk
from nltk.tokenize import sent_tokenize

def clean_text(text):
    """
    Performs basic cleaning of text by replacing specific markers and stripping whitespace.
    """
    # Replace <nl> with actual newline characters
    text = text.replace('<nl>', '\n')
    # Remove <sos> and <eos> markers if present
    text = text.replace('<sos>', '').replace('<eos>', '')
    text = ' '.join(word for word in text.split() if re.match("^[a-zA-Z0-9.,!?;:'\"-]+$", word))
    return text.strip()

def split_story(story, ratio=0.25):
    sentences = sent_tokenize(story)
    split_index = int(len(sentences) * ratio)
    
    # Ensure at least one sentence is in the input part and one in the target part
    split_index = max(1, min(len(sentences) - 1, split_index))
    
    input_text = ' '.join(sentences[:split_index]).strip()
    target_text = ' '.join(sentences[split_index:]).strip()
    return input_text, target_text

def preprocess_file(file_path):
    nltk.download('punkt')
    data = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                cleaned_story = clean_text(line)
                if cleaned_story:  # If the story is not empty
                    input_text, target_text = split_story(cleaned_story)
                    data.append({'input': input_text, 'target': target_text})
    except Exception as e:
        print(f"Error processing file: {e}")

    return data

def main():
    # Example usage
    data_path = os.path.join("data", "stories3", "data.txt")
    preprocessed_stories = preprocess_file(data_path)
    if preprocessed_stories:
        print("Input Part:", preprocessed_stories[0]['input'])
        print('======================================')
        print("Target Part:", preprocessed_stories[0]['target'])
    else:
        print("No stories were processed.")

if __name__=='__main__':
    main()