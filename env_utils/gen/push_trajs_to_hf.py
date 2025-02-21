from huggingface_hub import HfApi
import os

HF_TOKEN = os.environ['HF_TOKEN']

repo_id = "bosungkim/long_alfred"
commit_message = "Uploading a new file"

from pathlib import Path

# Create an instance of the API
api = HfApi()

base_dir = os.getcwd() + "/dataset/new_trajectories"

# Walk through all subdirectories
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".json"):  # Process only JSON files
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, base_dir)  # Preserve directory structure

            filename = relative_path.split("/")[-1]
            floorplan = relative_path.split("/")[-2]

            # Upload file to Hugging Face Dataset Hub
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=floorplan + "/" + filename,  # Maintain directory structure in repo
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Uploading {relative_path}",
                token=HF_TOKEN,
            )
            print(relative_path)
