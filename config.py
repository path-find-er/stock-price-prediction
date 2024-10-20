
import torch

# Device Configuration
# DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# torch configs for NVIDIA GPUs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Directory Configuration
PROCESSED_DIR = 'data/processed'
RAW_DIR = 'data/raw'
CHECKPOINT_DIR = 'checkpoints'

# Data Processing Configuration
SEQUENCE_LENGTH = 32
BATCH_SIZE = 2048
UPDATE_SCALERS = False

# Model Configuration
INPUT_SIZE = 3
OUTPUT_SIZE = 1
HIDDEN_SIZE = SEQUENCE_LENGTH

# Training Configuration
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
PATIENCE = 25  # for early stopping

# Evaluation Configuration
SAMPLE_RATIO = 1.0  # for calculating random guess stats

# Visualization Configuration
MOVING_AVERAGE_WINDOW = 30  # for price trend visualization

# Add any other configuration parameters here

# Function to get all config parameters as a dictionary
def get_config_dict():
    return {key: value for key, value in globals().items() 
            if key.isupper() and not key.startswith('__')}

# Print config data
print("Device:", DEVICE)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN benchmark:", torch.backends.cudnn.benchmark)
print("cuDNN deterministic:", torch.backends.cudnn.deterministic)
print("Processed directory:", PROCESSED_DIR)
print("Raw directory:", RAW_DIR)
print("Checkpoint directory:", CHECKPOINT_DIR)
print("Sequence length:", SEQUENCE_LENGTH)
print("Batch size:", BATCH_SIZE)
print("Update scalers:", UPDATE_SCALERS)
print("Input size:", INPUT_SIZE)
print("Output size:", OUTPUT_SIZE)
print("Hidden size:", HIDDEN_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Number of epochs:", NUM_EPOCHS)
print("Patience:", PATIENCE)
print("Sample ratio:", SAMPLE_RATIO)
print("Moving average window:", MOVING_AVERAGE_WINDOW)

# You can add more complex configurations or conditional logic here if needed
# For example:
# if DEVICE.type == 'cuda':
#     BATCH_SIZE *= 2