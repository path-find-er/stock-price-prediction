# This file was automatically generatedimport torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class PriceVolumeDataset(Dataset):
    def __init__(self, csv_file, sequence_length, scalers):
        self.df = pd.read_csv(csv_file, header=None)
        self.sequence_length = sequence_length
        self.scalers = scalers
        self.data = self._preprocess_data()

    def _preprocess_data(self):
        processed_data = []
        for _, row in self.df.iterrows():
            price, volume, datetime_int = self._series_to_list(row)
            if len(price) >= self.sequence_length:
                scaled_price = self.scalers['price'].transform([[p] for p in price]).flatten()
                scaled_volume = self.scalers['volume'].transform([[v] for v in volume]).flatten()
                scaled_datetime = self.scalers['datetime'].transform([[d] for d in datetime_int]).flatten()
                
                for i in range(len(price) - self.sequence_length):
                    sequence = np.column_stack((
                        scaled_price[i:i+self.sequence_length],
                        scaled_volume[i:i+self.sequence_length],
                        scaled_datetime[i:i+self.sequence_length]
                    ))
                    target = 1 if price[i+self.sequence_length] > price[i+self.sequence_length-1] else 0
                    processed_data.append((sequence, target))
        return processed_data

    def _series_to_list(self, series):
        price, volume, datetime_int = [], [], []
        for element in series:
            if pd.notna(element):
                p, v, d = element.split('|')
                price.append(float(p))
                volume.append(float(v))
                datetime_int.append(float(d))
        return price, volume, datetime_int

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, target = self.data[idx]
        return torch.FloatTensor(sequence), torch.FloatTensor([target])