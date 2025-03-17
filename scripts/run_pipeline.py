#!/usr/bin/env python
import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add the project root directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

def run_command(cmd, description=None):
    """Run a command and print its output in real-time."""
    if description:
        print(f"\n{'-' * 80}")
        print(f"⏳ {description}")
        print(f"{'-' * 80}\n")
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=project_root  # Run commands from the project root
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line.strip())
    
    # Wait for the process to complete and get the return code
    return_code = process.wait()
    
    if return_code == 0:
        if description:
            print(f"\n✅ {description} completed successfully.\n")
        return True
    else:
        if description:
            print(f"\n❌ {description} failed with return code {return_code}.\n")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the full Vietnamese VQA pipeline")
    parser.add_argument("--data-dir", type=str, default="./dummy_data",
                      help="Directory to store the dummy dataset")
    parser.add_argument("--checkpoint-dir", type=str, default="./vqa_checkpoints",
                      help="Directory to save models and results")
    parser.add_argument("--num-images", type=int, default=50,
                      help="Number of dummy images to generate")
    parser.add_argument("--num-questions", type=int, default=200,
                      help="Number of dummy questions to generate")
    parser.add_argument("--skip-dataset", action="store_true",
                      help="Skip dataset creation if it already exists")
    parser.add_argument("--epochs", type=int, default=3,
                      help="Number of epochs for training")
    parser.add_argument("--quick-test", action="store_true",
                      help="Run a quick test with minimal evaluation")
    parser.add_argument("--test-samples", type=int, default=10,
                      help="Number of test samples to evaluate")
    parser.add_argument("--skip-test", action="store_true",
                      help="Skip the testing step")
    parser.add_argument("--skip-train", action="store_true",
                      help="Skip the training step")
    
    args = parser.parse_args()
    
    # Make paths absolute
    data_dir = os.path.abspath(args.data_dir)
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    
    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Ensure subdirectories exist
    os.makedirs(os.path.join(data_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "ViVQA-csv"), exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, "test_results"), exist_ok=True)
    
    # Step 1: Create dummy dataset if needed
    if args.skip_dataset and os.path.exists(os.path.join(data_dir, "ViVQA-csv", "train.csv")):
        print(f"Skipping dataset creation as --skip-dataset is set and dataset exists at {data_dir}")
    else:
        # Use proper quoting for paths with spaces
        python_path = sys.executable
        script_path = os.path.join(script_dir, 'create_dummy_dataset.py')
        dataset_cmd = f'"{python_path}" "{script_path}" --output-dir "{data_dir}" --num-images {args.num_images} --num-questions {args.num_questions}'
        success = run_command(dataset_cmd, "Creating dummy dataset")
        if not success:
            print("Failed to create dataset. Exiting.")
            return 1
    
    # Check if dataset was properly created
    required_files = [
        os.path.join(data_dir, "vocab.json"),
        os.path.join(data_dir, "ViVQA-csv", "train.csv"),
        os.path.join(data_dir, "ViVQA-csv", "val.csv"),
        os.path.join(data_dir, "ViVQA-csv", "test.csv")
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file not found: {file_path}")
            print("Please check the dataset creation process or create the dataset manually.")
            return 1
    
    # Step 2: Test the setup using the dummy data
    if args.skip_test:
        print("Skipping test step as --skip-test is set")
    else:
        # Use proper quoting for paths with spaces
        script_path = os.path.join(script_dir, 'test_selective_vqa.py')
        quick_test_option = "--quick-test" if args.quick_test else ""
        test_cmd = (f'"{python_path}" "{script_path}" '
                   f'--data-dir "{data_dir}" '
                   f'--checkpoint-dir "{checkpoint_dir}" '
                   f'--num-samples {args.test_samples} '
                   f'{quick_test_option}')
        success = run_command(test_cmd, "Testing Selective VQA setup")
        if not success:
            print("Failed to test the setup. Exiting.")
            return 1
    
    # Step 3: Train the model
    if args.skip_train:
        print("Skipping training step as --skip-train is set")
    else:
        # Use proper quoting for paths with spaces
        script_path = os.path.join(script_dir, 'train_selective_vqa.py')
        images_path = os.path.join(data_dir, 'images')
        vocab_path = os.path.join(data_dir, 'vocab.json')
        train_path = os.path.join(data_dir, 'ViVQA-csv', 'train.csv')
        val_path = os.path.join(data_dir, 'ViVQA-csv', 'val.csv')
        test_path = os.path.join(data_dir, 'ViVQA-csv', 'test.csv')
        
        train_cmd = (f'"{python_path}" "{script_path}" '
                    f'--image-path "{images_path}" '
                    f'--ans-path "{vocab_path}" '
                    f'--train-path "{train_path}" '
                    f'--val-path "{val_path}" '
                    f'--test-path "{test_path}" '
                    f'--checkpoint-dir "{checkpoint_dir}" '
                    f'--epochs {args.epochs}')
        
        success = run_command(train_cmd, "Training Selective VQA model")
        if not success:
            print("Failed to train the model. Exiting.")
            return 1
    
    print("\n" + "=" * 80)
    print("🎉 Pipeline completed successfully!")
    print("=" * 80)
    print(f"• Dataset saved to: {data_dir}")
    print(f"• Model checkpoints and results saved to: {checkpoint_dir}")
    print("\nTo use the trained model, you can:")
    print(f"1. Run inference using the test script: python {os.path.join(script_dir, 'test_selective_vqa.py')} --data-dir {data_dir} --checkpoint-dir {checkpoint_dir} --load-model-only")
    print(f"2. Import the model in your code for inference")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 