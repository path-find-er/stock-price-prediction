import torch
import logging
import os

# Device Configuration
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# torch configs for NVIDIA GPUs
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False


# Directory Configuration
PROCESSED_DIR = "data/processed"
RAW_DIR = "data/raw"
CHECKPOINT_DIR = "checkpoints"

# Data Processing Configuration
SEQUENCE_LENGTH = 8
BATCH_SIZE = 2048 * 6
UPDATE_SCALERS = False

# Dataset Configuration
TRAIN_NUM_WORKERS = 8
VAL_NUM_WORKERS = 2
PREFETCH_FACTOR = 2
PIN_MEMORY = True


# Model Configuration
INPUT_SIZE = 3
OUTPUT_SIZE = 1
HIDDEN_SIZE = SEQUENCE_LENGTH * (SEQUENCE_LENGTH // 2)  # simulated attention

# Training Configuration
LEARNING_RATE = 0.001
NUM_EPOCHS = 5000
PATIENCE = 25  # for early stopping

# Evaluation Configuration
SAMPLE_RATIO = 1.0  # for calculating random guess stats

# Visualization Configuration
MOVING_AVERAGE_WINDOW = 30  # for price trend visualization


# Function to get all config parameters as a dictionary
def get_config_dict():
    return {
        key: value
        for key, value in globals().items()
        if key.isupper() and not key.startswith("__")
    }

# Function to print config data
def print_config():
    logging.info("Device: %s", DEVICE)
    logging.info("CUDA available: %s", torch.cuda.is_available())
    logging.info("cuDNN benchmark: %s", torch.backends.cudnn.benchmark)
    logging.info("cuDNN deterministic: %s", torch.backends.cudnn.deterministic)
    logging.info("Processed directory: %s", PROCESSED_DIR)
    logging.info("Raw directory: %s", RAW_DIR)
    logging.info("Checkpoint directory: %s", CHECKPOINT_DIR)
    logging.info("Sequence length: %d", SEQUENCE_LENGTH)
    logging.info("Batch size: %d", BATCH_SIZE)
    logging.info("Update scalers: %s", UPDATE_SCALERS)
    logging.info("Input size: %d", INPUT_SIZE)
    logging.info("Output size: %d", OUTPUT_SIZE)
    logging.info("Hidden size: %d", HIDDEN_SIZE)
    logging.info("Learning rate: %f", LEARNING_RATE)
    logging.info("Number of epochs: %d", NUM_EPOCHS)
    logging.info("Patience: %d", PATIENCE)
    logging.info("Sample ratio: %f", SAMPLE_RATIO)
    logging.info("Moving average window: %d", MOVING_AVERAGE_WINDOW)


# Add more complex configurations or conditional logic here if needed
# For example:
# if DEVICE.type == 'cuda':
#     BATCH_SIZE *= 2

# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)

# Logger Configuration
# set the name based on config data
log_filename = f'logs/app_{DEVICE.type}_{SEQUENCE_LENGTH}_{BATCH_SIZE}_{HIDDEN_SIZE}_{LEARNING_RATE:.6f}.log'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_filename,
    filemode='a'
)
logger = logging.getLogger(__name__)

# Add a stream handler to also print logs to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)