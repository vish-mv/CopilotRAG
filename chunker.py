import os
import re


def chunk_markdown(content):
    chunks = []
    current_chunk = []
    lines = content.split('\n')

    for line in lines:
        if line.startswith('# ') or line.startswith('## '):
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
        current_chunk.append(line)

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks


def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.md'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            chunks = chunk_markdown(content)

            print(f"File: {filename}")
            for i, chunk in enumerate(chunks, 1):
                print(f"Chunk {i}:")
                print(chunk)
                print("-" * 40)
            print("=" * 40)


# Replace 'path/to/your/folder' with the actual path to your folder containing .md files
