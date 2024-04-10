import os
import multiprocessing
from functools import partial

def clean_text(text):
    """
    Performs basic cleaning of text by unescaping internal quotes and stripping whitespace.
    """
    text = text.replace('""', '"')
    text = text.strip()
    return text

def process_line(line, in_story, story_lines):
    if line.startswith('"') and not in_story:
        in_story = True
        story_lines.append(line[1:].strip())
    elif line.endswith('"\n') and in_story:
        story_lines.append(line[:-2].strip())
        full_story = " ".join(story_lines)
        full_story_cleaned = clean_text(full_story)
        
        # Find the split point that does not cut off mid-word
        split_point = len(full_story_cleaned) // 4
        # Adjust the split point to the nearest space to avoid splitting words
        while split_point < len(full_story_cleaned) and full_story_cleaned[split_point] not in [' ', '\n']:
            split_point += 1
        
        story = {
            'input': full_story_cleaned[:split_point].strip(),
            'target': full_story_cleaned[split_point:].strip()
        }
        story_lines = []
        in_story = False
        return story, in_story, story_lines
    elif in_story:
        story_lines.append(line.strip())
    return None, in_story, story_lines

def process_chunk(file_path, start, end):
    stories = []
    with open(file_path, 'r', encoding='utf-8') as file:
        if start != 0:
            file.seek(start)
            file.readline()
        in_story = False
        story_lines = []
        while file.tell() < end:
            line = file.readline()
            story, in_story, story_lines = process_line(line, in_story, story_lines)
            if story:
                stories.append(story)
    return stories

def chunkify(file_path, size=1024*1024*50):
    file_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as file:
        chunk_ends = [0]
        while True:
            start = file.tell()
            file.seek(size, 1)
            file.readline()
            end = file.tell()
            if end >= file_size:
                chunk_ends.append(file_size)
                break
            else:
                chunk_ends.append(end)
    return chunk_ends

def parallel_process_file(file_path):
    chunk_ends = chunkify(file_path)
    chunk_starts = chunk_ends[:-1]
    chunk_ends = chunk_ends[1:]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        func = partial(process_chunk, file_path)
        results = pool.starmap(func, zip(chunk_starts, chunk_ends))

    stories = [story for chunk in results for story in chunk]
    return stories

def main():
    data_path = os.path.join("data", "stories", "validation.csv")
    processed_stories = parallel_process_file(data_path)
    for story in processed_stories[:4]:
        print(story)

if __name__=='__main__':
    main()