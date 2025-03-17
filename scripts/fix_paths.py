#!/usr/bin/env python
import os
import sys
import re
import glob

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

def find_python_files(directory):
    """Find all Python files in the given directory and its subdirectories"""
    return glob.glob(os.path.join(directory, "**", "*.py"), recursive=True)

def fix_file_opening(content):
    """Fix file opening operations to include UTF-8 encoding"""
    # Pattern to match file opening without encoding
    pattern = r"open\(([^,]+),\s*['\"]r['\"]\)"
    replacement = r"open(\1, 'r', encoding='utf-8')"
    
    # Apply the replacement
    fixed_content = re.sub(pattern, replacement, content)
    
    # Pattern for JSON writes without encoding
    write_pattern = r"open\(([^,]+),\s*['\"]w['\"]\)"
    write_replacement = r"open(\1, 'w', encoding='utf-8')"
    
    # Apply the write replacement
    fixed_content = re.sub(write_pattern, write_replacement, fixed_content)
    
    return fixed_content

def fix_file_paths(file_path):
    """Fix file paths in a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix file opening operations
        fixed_content = fix_file_opening(content)
        
        # Check if the content was modified
        if fixed_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"✅ Fixed file: {file_path}")
        else:
            print(f"✓ No changes needed: {file_path}")
            
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            
            # Fix file opening operations
            fixed_content = fix_file_opening(content)
            
            # Save with UTF-8 encoding
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"✅ Fixed file with encoding change: {file_path}")
        except Exception as e:
            print(f"❌ Failed to process file: {file_path} - {str(e)}")
    except Exception as e:
        print(f"❌ Failed to process file: {file_path} - {str(e)}")

def main():
    """Main function to find and fix file paths in Python files"""
    python_files = find_python_files(project_root)
    print(f"Found {len(python_files)} Python files to check")
    
    for file_path in python_files:
        # Skip this script itself
        if os.path.samefile(file_path, __file__):
            continue
        fix_file_paths(file_path)
    
    print("\nPath fixing completed!")
    print("You can now run your scripts with proper file path handling.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 