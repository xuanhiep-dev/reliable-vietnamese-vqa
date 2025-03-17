#!/usr/bin/env python
import os
import sys

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import the create_dummy_dataset function
from scripts.create_dummy_dataset import create_dummy_dataset

def main():
    # Set default paths
    output_dir = os.path.join(project_root, "dummy_data")
    
    # Create the dataset
    print(f"Creating dummy dataset in {output_dir}")
    create_dummy_dataset(output_dir, num_images=50, num_questions=200)
    
    # Verify files were created
    required_files = [
        os.path.join(output_dir, "vocab.json"),
        os.path.join(output_dir, "ViVQA-csv", "train.csv"),
        os.path.join(output_dir, "ViVQA-csv", "val.csv"),
        os.path.join(output_dir, "ViVQA-csv", "test.csv")
    ]
    
    all_exist = True
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file not found: {file_path}")
            all_exist = False
    
    if all_exist:
        print("✅ All required files were created successfully!")
        print(f"\nYou can now run the test script with:")
        print(f"python scripts/test_selective_vqa.py --data-dir {output_dir} --checkpoint-dir ./vqa_checkpoints")
    else:
        print("❌ Some files are missing. The dataset creation may have failed.")
    
    return 0 if all_exist else 1

if __name__ == "__main__":
    sys.exit(main()) 