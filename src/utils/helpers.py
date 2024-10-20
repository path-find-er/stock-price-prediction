import math
import torch
from torch.utils.data import DataLoader
import pandas as pd
import logging


def calculate_random_guess_stats(dataset, num_samples: int = None) -> dict[str, float]:
    dataloader: DataLoader = DataLoader(dataset, batch_size=1)
    total_samples: int = 0
    positive_samples: int = 0

    for _, target in dataloader:
        total_samples += 1
        if target.item() == 1:
            positive_samples += 1

        if num_samples is not None and total_samples >= num_samples:
            break

    # Calculate the actual proportion of positive and negative samples
    actual_positive_ratio: float = positive_samples / total_samples
    actual_negative_ratio: float = 1 - actual_positive_ratio

    # Expected accuracy when guessing according to class probabilities
    accuracy: float = actual_positive_ratio**2 + actual_negative_ratio**2

    # Calculate the standard error of the accuracy
    std_error: float = math.sqrt((accuracy * (1 - accuracy)) / total_samples)

    return {
        "total_samples": total_samples,
        "expected_accuracy": accuracy,
        "std_error": std_error,
        "actual_positive_ratio": actual_positive_ratio,
        "actual_negative_ratio": actual_negative_ratio,
    }


def series_to_list(series: pd.Series) -> tuple[list[float], list[float], list[float]]:
    price: list[float] = []
    volume: list[float] = []
    datetime_int: list[float] = []
    for element in series:
        if pd.notna(element):
            p, v, d = element.split("|")
            price.append(float(p))
            volume.append(float(v))
            datetime_int.append(float(d))
    return price, volume, datetime_int


def estimate_iterations(
    csv_file: str, sequence_length: int, batch_size: int, sample_ratio: float = 1.0
) -> int:
    df: pd.DataFrame = pd.read_csv(csv_file, header=None).sample(frac=sample_ratio)
    total_sequences: int = sum(
        max(0, len(row.dropna()) - sequence_length + 1) for _, row in df.iterrows()
    )
    estimated_iterations: int = math.ceil(total_sequences / batch_size)
    logging.info(f"Total sequences: {total_sequences}")
    logging.info(f"Estimated iterations per epoch: {estimated_iterations}")
    return estimated_iterations
