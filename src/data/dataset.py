import math
import torch
from torch.utils.data import IterableDataset, DataLoader
import csv
from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy as np

class PriceVolumeDataset(IterableDataset):
    def __init__(self, csv_file, sequence_length, scalers=None):
        self.csv_file = csv_file
        self.sequence_length = sequence_length
        self.scalers = scalers

    def __iter__(self):
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)
            buffer = []
            for line in reader:
                for cell in line:
                    data = cell.split('|')
                    if len(data) != 3:
                        continue
                    price, volume, datetime_int = data
                    if not all(map(self.is_valid_float, [price, volume, datetime_int])):
                        continue
                    price, volume, datetime_int = map(float, [price, volume, datetime_int])
                    if self.scalers:
                        price = self.scalers['price'].transform([[price]])[0][0]
                        volume = self.scalers['volume'].transform([[volume]])[0][0]
                        datetime_int = self.scalers['datetime'].transform([[datetime_int]])[0][0]

                    buffer.append([price, volume, datetime_int])
                    
                    # When buffer reaches sequence_length + 1, yield the sequence and target
                    if len(buffer) == self.sequence_length + 1:
                        sequence = torch.tensor(buffer[:-1], dtype=torch.float32)
                        # Set target to 1 if price went up, 0 if it stayed the same or went down
                        target = torch.tensor(1.0 if buffer[-1][0] > buffer[-2][0] else 0.0, dtype=torch.float32).unsqueeze(0)
                        yield sequence, target
                        # Remove the oldest element to make room for the next one
                        buffer.pop(0)

    @staticmethod
    def is_valid_float(value):
        try:
            float_value = float(value)
            return not np.isnan(float_value) and not np.isinf(float_value)
        except ValueError:
            return False