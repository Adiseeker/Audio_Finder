import os

# Open the lista.txt file and read the file paths
with open('D:\\Ai\\Audio-Classifier\\voiceapp\\lista.txt', 'r') as f:
    file_paths = [line.strip() for line in f.readlines()]

# Replace the .mp3 extension with .txt for each file path
txt_files = [file_path.replace('.mp3', '.txt') for file_path in file_paths]



# Merge the contents of the .txt files into a single string
merged_content = ''
for file_path in txt_files:
    with open(file_path, 'r') as f:
        print(merged_content)
        merged_content += f.read()


# Save the merged content to a new file called vault.txt
with open('D:\\Ai\\Audio-Classifier\\voiceapp\\vault.txt', 'w') as f:
    f.write(merged_content)