import os

def clean_text(text):
    """
    Performs basic cleaning of text by replacing specific markers and stripping whitespace.
    """
    # Replace <nl> with actual newline characters
    text = text.replace('<nl>', '\n')
    # Remove <sos> and <eos> markers if present
    text = text.replace('<sos>', '').replace('<eos>', '')
    return text.strip()

def split_story(story):
    split_index = len(story) // 4
    # Ensure the split happens at the end of a sentence where possible
    while split_index < len(story) and story[split_index] not in ".!?":
        split_index += 1
    # Adjust split_index to include the punctuation mark
    split_index += 1
    return story[:split_index].strip(), story[split_index:].strip()

def preprocess_file(file_path):
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