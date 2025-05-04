import os
import json
import shutil
from tqdm import tqdm
def create_training_data(jsonl_file, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get the root directory of the JSONLines file
    jsonl_root = os.path.dirname(jsonl_file)

    with open(jsonl_file, 'r') as file:
        for index, line in tqdm(enumerate(file)):
            data = json.loads(line.strip())
            file_name = os.path.join(jsonl_root, data["file_name"])  # Prepend root directory
            text = data["text"]

            # Define new file names
            new_image_name = f"image_{index}.jpg"
            new_json_name = f"image_{index}.json"

            # Copy the image to the output folder
            shutil.copy(file_name, os.path.join(output_folder, new_image_name))

            # Create the JSON file with the required structure
            json_content = {
                "prompt": text,
                "generated_prompt": None  # Placeholder for generated caption
            }
            with open(os.path.join(output_folder, new_json_name), 'w') as json_file:
                json.dump(json_content, json_file, indent=4)

if __name__ == "__main__":
    # Input JSONLines file and output folder
    input_jsonl = "/leonardo_scratch/large/userexternal/lsigillo/datasets--zhang0jhon--Aesthetic-4K/snapshots/40a07c883a5c824a42b3dac2e393f705579cbf82/eval/size_2048/metadata.jsonl"  # Replace with the actual path
    output_dir = "/leonardo_scratch/large/userexternal/lsigillo/Aesthetic-4K/eval/size_2048/"  # Replace with the desired output folder name

    create_training_data(input_jsonl, output_dir)
