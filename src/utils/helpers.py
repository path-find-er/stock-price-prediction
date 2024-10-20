import math
import torch
from torch.utils.data import DataLoader
import pandas as pd

def calculate_random_guess_stats(dataset, num_samples=None):
    dataloader = DataLoader(dataset, batch_size=1)
    total_samples = 0
    positive_samples = 0

    for _, target in dataloader:
        total_samples += 1
        if target.item() == 1:
            positive_samples += 1
            
        if num_samples is not None and total_samples >= num_samples:
            break

    # Calculate the actual proportion of positive and negative samples
    actual_positive_ratio = positive_samples / total_samples
    actual_negative_ratio = 1 - actual_positive_ratio

    # Expected accuracy when guessing according to class probabilities
    accuracy = actual_positive_ratio**2 + actual_negative_ratio**2

    # Calculate the standard error of the accuracy
    std_error = math.sqrt((accuracy * (1 - accuracy)) / total_samples)

    return {
        'total_samples': total_samples,
        'expected_accuracy': accuracy,
        'std_error': std_error,
        'actual_positive_ratio': actual_positive_ratio,
        'actual_negative_ratio': actual_negative_ratio
    }

def series_to_list(series):
    price, volume, datetime_int = [], [], []
    for element in series:
        if pd.notna(element):
            p, v, d = element.split('|')
            price.append(float(p))
            volume.append(float(v))
            datetime_int.append(float(d))
    return price, volume, datetime_int

def estimate_iterations(csv_file, sequence_length, batch_size, sample_ratio=1.0):
    df = pd.read_csv(csv_file, header=None).sample(frac=sample_ratio)
    total_sequences = sum(max(0, len(row.dropna()) - sequence_length + 1) for _, row in df.iterrows())
    estimated_iterations = math.ceil(total_sequences / batch_size)
    print(f"Total sequences: {total_sequences}")
    print(f"Estimated iterations per epoch: {estimated_iterations}")
    return estimated_iterations