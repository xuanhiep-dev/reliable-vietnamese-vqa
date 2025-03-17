#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
DATA_DIR="./dummy_data"
CHECKPOINT_DIR="./vqa_checkpoints"
NUM_IMAGES=50
NUM_QUESTIONS=200
EPOCHS=3
TEST_SAMPLES=10
SKIP_DATASET=false
SKIP_TRAINING=false
SKIP_TESTING=false
MINIMAL_MODEL=false
QUICK_EVAL=false

# Function to display usage information
show_usage() {
    echo -e "${BLUE}Usage:${NC} $0 [OPTIONS]"
    echo -e "Runs the complete Vietnamese Reliable VQA pipeline."
    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  --data-dir DIR         Directory to store dataset (default: $DATA_DIR)"
    echo "  --checkpoint-dir DIR   Directory to save models (default: $CHECKPOINT_DIR)"
    echo "  --num-images NUM       Number of dummy images to generate (default: $NUM_IMAGES)"
    echo "  --num-questions NUM    Number of dummy questions to generate (default: $NUM_QUESTIONS)"
    echo "  --epochs NUM           Number of epochs for training (default: $EPOCHS)"
    echo "  --test-samples NUM     Number of samples for testing (default: $TEST_SAMPLES)"
    echo "  --skip-dataset         Skip dataset creation"
    echo "  --skip-training        Skip model training"
    echo "  --skip-testing         Skip model testing"
    echo "  --minimal-model        Use minimal model for faster execution"
    echo "  --quick-eval           Run quick evaluation with fewer samples"
    echo "  --help                 Show this help message"
    echo ""
    echo -e "${BLUE}Example:${NC}"
    echo "  $0 --data-dir ./my_data --checkpoint-dir ./my_checkpoints --epochs 5"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --num-images)
            NUM_IMAGES="$2"
            shift 2
            ;;
        --num-questions)
            NUM_QUESTIONS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --test-samples)
            TEST_SAMPLES="$2"
            shift 2
            ;;
        --skip-dataset)
            SKIP_DATASET=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-testing)
            SKIP_TESTING=true
            shift
            ;;
        --minimal-model)
            MINIMAL_MODEL=true
            shift
            ;;
        --quick-eval)
            QUICK_EVAL=true
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

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$DATA_DIR/images"
mkdir -p "$DATA_DIR/ViVQA-csv"

# Function to print section header
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

# Function to run a command with error handling
run_command() {
    echo -e "${YELLOW}Running:${NC} $1"
    if eval "$1"; then
        echo -e "${GREEN}Command completed successfully!${NC}"
        return 0
    else
        echo -e "${RED}Command failed with exit code $?${NC}"
        return 1
    fi
}

# Step 1: Create dataset
if [ "$SKIP_DATASET" = true ]; then
    print_header "SKIPPING DATASET CREATION (--skip-dataset specified)"
else
    print_header "STEP 1: CREATING DATASET"
    
    # Use direct_create_dataset.py
    DATASET_CMD="python scripts/direct_create_dataset.py --output-dir \"$DATA_DIR\" --num-images $NUM_IMAGES --num-questions $NUM_QUESTIONS"
    
    if ! run_command "$DATASET_CMD"; then
        echo -e "${RED}Dataset creation failed. Exiting pipeline.${NC}"
        exit 1
    fi
fi

# Check if dataset exists
if [ ! -f "$DATA_DIR/vocab.json" ] || [ ! -f "$DATA_DIR/ViVQA-csv/train.csv" ]; then
    echo -e "${RED}Required dataset files not found. Please check dataset creation or specify correct --data-dir.${NC}"
    exit 1
fi

# Step 2: Train model
if [ "$SKIP_TRAINING" = true ]; then
    print_header "SKIPPING MODEL TRAINING (--skip-training specified)"
else
    print_header "STEP 2: TRAINING SELECTIVE VQA MODEL"
    
    # Build the command with appropriate flags
    TRAIN_CMD="python scripts/train_selective_vqa.py \
        --image-path \"$DATA_DIR/images\" \
        --ans-path \"$DATA_DIR/vocab.json\" \
        --train-path \"$DATA_DIR/ViVQA-csv/train.csv\" \
        --val-path \"$DATA_DIR/ViVQA-csv/val.csv\" \
        --test-path \"$DATA_DIR/ViVQA-csv/test.csv\" \
        --checkpoint-dir \"$CHECKPOINT_DIR\" \
        --epochs $EPOCHS"
    
    # Add optional flags
    if [ "$MINIMAL_MODEL" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --minimal-model"
    fi
    
    if [ "$QUICK_EVAL" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --quick-eval --eval-samples $TEST_SAMPLES"
    fi
    
    if ! run_command "$TRAIN_CMD"; then
        echo -e "${YELLOW}Warning: Training may have encountered issues, but continuing with pipeline.${NC}"
    fi
fi

# Step 3: Test model
if [ "$SKIP_TESTING" = true ]; then
    print_header "SKIPPING MODEL TESTING (--skip-testing specified)"
else
    print_header "STEP 3: TESTING SELECTIVE VQA MODEL"
    
    # Build the command with appropriate flags
    TEST_CMD="python scripts/test_selective_vqa.py \
        --data-dir \"$DATA_DIR\" \
        --checkpoint-dir \"$CHECKPOINT_DIR\" \
        --num-samples $TEST_SAMPLES"
    
    # Add optional flags
    if [ "$QUICK_EVAL" = true ] || [ "$MINIMAL_MODEL" = true ]; then
        TEST_CMD="$TEST_CMD --quick-test"
    fi
    
    if ! run_command "$TEST_CMD"; then
        echo -e "${YELLOW}Warning: Testing encountered issues.${NC}"
    fi
fi

# Pipeline complete
print_header "PIPELINE COMPLETE"
echo -e "${GREEN}The Vietnamese Reliable VQA pipeline has finished running.${NC}"
echo -e "Dataset location: ${BLUE}$DATA_DIR${NC}"
echo -e "Model checkpoints: ${BLUE}$CHECKPOINT_DIR${NC}"
echo -e "Test results: ${BLUE}$CHECKPOINT_DIR/test_results${NC}"

echo -e "\n${YELLOW}To use the trained model for predictions:${NC}"
echo "python scripts/test_selective_vqa.py --data-dir \"$DATA_DIR\" --checkpoint-dir \"$CHECKPOINT_DIR\" --skip-model-creation"

exit 0 