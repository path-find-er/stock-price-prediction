
import torch

# Device Configuration
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

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

# You can add more complex configurations or conditional logic here if needed
# For example:
# if DEVICE.type == 'cuda':
#     BATCH_SIZE *= 2