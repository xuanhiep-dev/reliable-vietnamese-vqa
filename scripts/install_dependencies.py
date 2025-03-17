#!/usr/bin/env python
import os
import sys
import subprocess
import pkg_resources

def run_command(cmd):
    """Run a command and print its output."""
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    return process.returncode

def check_package(package_name):
    """Check if a package is installed."""
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False

def install_dependencies():
    """Install required dependencies for the Vietnamese-Reliable-VQA project."""
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Read dependencies from requirements.txt
    requirements_path = os.path.join(project_root, "requirements.txt")
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = f.read().splitlines()
    
    # Filter out empty lines and comments
    requirements = [r for r in requirements if r and not r.startswith("#")]
    
    print("Checking and installing dependencies...")
    
    # Install each dependency
    for req in requirements:
        # Extract package name (before any version specifier)
        pkg_name = req.split('==')[0].split('@')[0].split('>')[0].split('<')[0].strip()
        
        if not check_package(pkg_name):
            print(f"Installing {req}...")
            result = run_command(f"pip install {req}")
            
            if result != 0:
                print(f"Failed to install {req}")
                if pkg_name == "torchscale":
                    print("Trying to install torchscale from GitHub...")
                    run_command("pip install git+https://github.com/microsoft/torchscale.git")
            else:
                print(f"Successfully installed {req}")
        else:
            print(f"{pkg_name} is already installed")
    
    print("\nDependencies check complete!")

if __name__ == "__main__":
    install_dependencies() 