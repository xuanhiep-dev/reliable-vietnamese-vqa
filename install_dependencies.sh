#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
CREATE_VENV=false
VENV_NAME="vqa_env"
CUDA_VERSION="118"  # Default CUDA version 11.8
FORCE_REINSTALL=false
CPU_ONLY=false

# Function to display usage information
show_usage() {
    echo -e "${BLUE}Usage:${NC} $0 [OPTIONS]"
    echo -e "Installs all dependencies required for the Vietnamese Reliable VQA project."
    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  --venv                 Create and use a virtual environment"
    echo "  --venv-name NAME       Set virtual environment name (default: $VENV_NAME)"
    echo "  --cuda-version VER     Set CUDA version (112, 116, 117, 118, 121) (default: $CUDA_VERSION)"
    echo "  --cpu-only             Install CPU-only versions of PyTorch and dependencies"
    echo "  --force-reinstall      Force reinstall all packages"
    echo "  --help                 Show this help message"
    echo ""
    echo -e "${BLUE}Example:${NC}"
    echo "  $0 --venv --cuda-version 118"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
            CREATE_VENV=true
            shift
            ;;
        --venv-name)
            VENV_NAME="$2"
            shift 2
            ;;
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --force-reinstall)
            FORCE_REINSTALL=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Function to print section header
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python 3 is installed
print_header "CHECKING PYTHON INSTALLATION"
if ! command_exists python3 && ! command_exists python; then
    echo -e "${RED}Error: Python 3 is not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Use python or python3 depending on what's available
PYTHON_CMD="python3"
if ! command_exists python3; then
    PYTHON_CMD="python"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "Python version: ${GREEN}$PYTHON_VERSION${NC}"

if [[ "${PYTHON_VERSION%%.*}" -lt 3 ]]; then
    echo -e "${RED}Error: Python 3 is required, but Python $PYTHON_VERSION is installed.${NC}"
    exit 1
fi

# Create virtual environment if requested
if [ "$CREATE_VENV" = true ]; then
    print_header "SETTING UP VIRTUAL ENVIRONMENT"
    
    # Check if virtualenv is installed
    if ! $PYTHON_CMD -m pip show virtualenv >/dev/null 2>&1; then
        echo "Installing virtualenv..."
        $PYTHON_CMD -m pip install --user virtualenv
    fi
    
    # Create and activate virtual environment
    echo "Creating virtual environment '$VENV_NAME'..."
    $PYTHON_CMD -m virtualenv "$VENV_NAME"
    
    # Determine activation script based on OS
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        # Windows
        ACTIVATE_SCRIPT="./$VENV_NAME/Scripts/activate"
    else
        # Linux, macOS
        ACTIVATE_SCRIPT="./$VENV_NAME/bin/activate"
    fi
    
    echo "Activating virtual environment..."
    if [[ -f "$ACTIVATE_SCRIPT" ]]; then
        source "$ACTIVATE_SCRIPT"
        echo -e "${GREEN}Virtual environment activated!${NC}"
        # Use python from venv
        PYTHON_CMD="python"
    else
        echo -e "${RED}Failed to activate virtual environment. Continuing with system Python...${NC}"
    fi
fi

# Upgrade pip
print_header "UPGRADING PIP"
$PYTHON_CMD -m pip install --upgrade pip

# Install PyTorch
print_header "INSTALLING PYTORCH"

# Determine PyTorch install command based on user preferences
if [ "$CPU_ONLY" = true ]; then
    TORCH_INSTALL="torch torchvision torchaudio"
    echo "Installing CPU-only PyTorch..."
else
    # Determine PyTorch CUDA version
    case "$CUDA_VERSION" in
        "112")
            TORCH_INSTALL="torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu112"
            ;;
        "116")
            TORCH_INSTALL="torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116"
            ;;
        "117")
            TORCH_INSTALL="torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117"
            ;;
        "118")
            TORCH_INSTALL="torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            ;;
        "121")
            TORCH_INSTALL="torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            ;;
        *)
            echo -e "${YELLOW}Warning: Unknown CUDA version '$CUDA_VERSION'. Using default (cu118).${NC}"
            TORCH_INSTALL="torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            ;;
    esac
    echo "Installing PyTorch with CUDA $CUDA_VERSION support..."
fi

# Install PyTorch
if [ "$FORCE_REINSTALL" = true ]; then
    $PYTHON_CMD -m pip install --force-reinstall $TORCH_INSTALL
else
    $PYTHON_CMD -m pip install $TORCH_INSTALL
fi

# Install specific version of OpenCV
print_header "INSTALLING SPECIFIC OPENCV VERSION"
$PYTHON_CMD -m pip install opencv-python==4.8.0.74

# Install dependencies from requirements.txt
print_header "INSTALLING DEPENDENCIES FROM REQUIREMENTS.txt"

INSTALL_OPTS=""
if [ "$FORCE_REINSTALL" = true ]; then
    INSTALL_OPTS="--force-reinstall"
fi

# Exclude torch and opencv which we've installed separately
$PYTHON_CMD -m pip install $INSTALL_OPTS -r requirements.txt --no-deps

# Install specific problematic packages separately
print_header "INSTALLING SPECIAL DEPENDENCIES"

# TorchScale installation
echo "Installing TorchScale..."
$PYTHON_CMD -m pip install git+https://github.com/microsoft/torchscale.git

# CLIP installation
echo "Installing CLIP from OpenAI..."
$PYTHON_CMD -m pip install git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33

# Install other dependencies
echo "Installing other required packages..."
$PYTHON_CMD -m pip install transformers timm matplotlib pandas numpy scikit-learn h5py tqdm Pillow mlflow efficientnet_pytorch

# Install underthesea for Vietnamese NLP
echo "Installing underthesea for Vietnamese NLP..."
$PYTHON_CMD -m pip install underthesea

# Install salesforce-lavis
echo "Installing salesforce-lavis..."
$PYTHON_CMD -m pip install salesforce-lavis

# Verify installations
print_header "VERIFYING INSTALLATIONS"

echo "Checking PyTorch installation..."
$PYTHON_CMD -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}')"

echo "Checking OpenCV installation..."
$PYTHON_CMD -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

echo "Checking TorchScale installation..."
$PYTHON_CMD -c "import torchscale; print('TorchScale is installed')" || echo -e "${YELLOW}Warning: TorchScale installation may have issues${NC}"

echo "Checking CLIP installation..."
$PYTHON_CMD -c "import clip; print('CLIP is installed')" || echo -e "${YELLOW}Warning: CLIP installation may have issues${NC}"

# Print success message
print_header "INSTALLATION COMPLETE"
echo -e "${GREEN}All dependencies have been installed successfully!${NC}"

if [ "$CREATE_VENV" = true ]; then
    echo -e "\n${YELLOW}To activate the virtual environment in future terminal sessions:${NC}"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo "source ./$VENV_NAME/Scripts/activate"
    else 
        echo "source ./$VENV_NAME/bin/activate"
    fi
fi

echo -e "\n${YELLOW}To run the pipeline:${NC}"
echo "./run_pipeline.sh"

exit 0 