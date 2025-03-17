#!/usr/bin/env python
import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import argparse
import shutil
import sys

# Add project root to path so we can import from modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def generate_dummy_image(output_path, width=224, height=224):
    """Generate a random color dummy image"""
    # Create a random color image
    r = np.random.randint(0, 255)
    g = np.random.randint(0, 255)
    b = np.random.randint(0, 255)
    
    # Create solid color image
    img = Image.new('RGB', (width, height), color=(r, g, b))
    img.save(output_path)

def create_dummy_images(image_dir, num_images=50):
    """Create dummy images for testing"""
    create_directory(image_dir)
    
    # Generate image IDs
    image_ids = [f"img_{i:04d}" for i in range(num_images)]
    
    for img_id in image_ids:
        image_path = os.path.join(image_dir, f"{img_id}.jpg")
        if not os.path.exists(image_path):
            generate_dummy_image(image_path)
            print(f"Created dummy image: {image_path}")
    
    return image_ids

def create_vocab_file(vocab_path):
    """Create a vocabulary file with common Vietnamese answers"""
    # Sample Vietnamese answers
    answers = [
        "có", "không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
        "trắng", "đen", "đỏ", "xanh", "vàng", "nâu", "cam", "tím", "hồng", "xám",
        "lớn", "nhỏ", "cao", "thấp", "rộng", "hẹp", "dài", "ngắn",
        "người", "chó", "mèo", "chim", "cá", "cây", "hoa", "quả", "xe", "nhà",
        "bàn", "ghế", "giường", "tủ", "cửa", "cửa sổ", "sách", "bút", "giấy", "máy tính"
    ]
    
    # Create vocabulary with answer indices
    vocab = {"answer": {answer: i for i, answer in enumerate(answers)}}
    
    # Write to file with UTF-8 encoding
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"Created vocabulary file: {vocab_path}")
    return answers

def create_csv_files(csv_dir, image_ids, vocab_answers, num_questions=200):
    """Create CSV files for training, validation and testing"""
    create_directory(csv_dir)
    
    # Simple Vietnamese question templates
    question_templates = [
        "Đây có phải là {}?",
        "Màu của vật thể trong ảnh là gì?",
        "Có bao nhiêu {} trong ảnh?",
        "Mô tả vật thể này là gì?",
        "Vật thể này có màu gì?",
        "Đây là gì?",
        "Vật này có phải là {} không?",
        "Bạn nhìn thấy gì trong ảnh?",
        "Có {} trong ảnh không?",
        "Màu sắc chủ đạo của ảnh là gì?"
    ]
    
    # Generate questions and answers
    questions = []
    answers = []
    img_ids = []
    
    for _ in range(num_questions):
        img_id = np.random.choice(image_ids)
        template = np.random.choice(question_templates)
        
        # For templates with placeholder, replace it with a random object
        if "{}" in template:
            object_word = np.random.choice(["người", "chó", "mèo", "xe", "nhà", "cây", "hoa"])
            question = template.format(object_word)
        else:
            question = template
        
        # Random answer from our vocabulary
        answer = np.random.choice(vocab_answers)
        
        questions.append(question)
        answers.append(answer)
        img_ids.append(img_id)
    
    # Create dataframe
    df = pd.DataFrame({
        'img_id': img_ids,
        'question': questions,
        'answer': answers
    })
    
    # Split into train, validation, and test sets (70%, 15%, 15%)
    n = len(df)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    df_train = df.iloc[:train_size]
    df_val = df.iloc[train_size:train_size+val_size]
    df_test = df.iloc[train_size+val_size:]
    
    # Save to CSV files
    train_path = os.path.join(csv_dir, 'train.csv')
    val_path = os.path.join(csv_dir, 'val.csv')
    test_path = os.path.join(csv_dir, 'test.csv')
    
    df_train.to_csv(train_path)
    df_val.to_csv(val_path)
    df_test.to_csv(test_path)
    
    print(f"Created CSV files in {csv_dir}:")
    print(f"  - Train: {train_path} ({len(df_train)} samples)")
    print(f"  - Validation: {val_path} ({len(df_val)} samples)")
    print(f"  - Test: {test_path} ({len(df_test)} samples)")

def create_dummy_dataset(output_dir, num_images=50, num_questions=200):
    """Create a complete dummy dataset for ViVQA"""
    # Create main directory
    create_directory(output_dir)
    
    # Create images directory and dummy images
    images_dir = os.path.join(output_dir, 'images')
    image_ids = create_dummy_images(images_dir, num_images)
    
    # Create vocabulary file
    vocab_path = os.path.join(output_dir, 'vocab.json')
    vocab_answers = create_vocab_file(vocab_path)
    
    # Create CSV directory and files
    csv_dir = os.path.join(output_dir, 'ViVQA-csv')
    create_csv_files(csv_dir, image_ids, vocab_answers, num_questions)
    
    print(f"\nDummy dataset created successfully at: {output_dir}")
    print(f"Number of images: {len(image_ids)}")
    print(f"Number of questions: {num_questions}")
    print(f"Number of possible answers: {len(vocab_answers)}")

def main():
    parser = argparse.ArgumentParser(description='Create dummy dataset for Vietnamese VQA')
    parser.add_argument('--output-dir', type=str, default='./dummy_data',
                        help='Directory to save the dummy dataset')
    parser.add_argument('--num-images', type=int, default=50,
                        help='Number of dummy images to generate')
    parser.add_argument('--num-questions', type=int, default=200,
                        help='Number of dummy questions to generate')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute for clarity in output
    args.output_dir = os.path.abspath(args.output_dir)
    
    print(f"Creating dummy dataset with:")
    print(f"- {args.num_images} images")
    print(f"- {args.num_questions} questions")
    print(f"- Output directory: {args.output_dir}")
    
    create_dummy_dataset(args.output_dir, args.num_images, args.num_questions)
    return 0

if __name__ == '__main__':
    main() 