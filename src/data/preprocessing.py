import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def series_to_list(series):
    price, volume, datetime_int = [], [], []
    for element in series:
        if pd.notna(element):
            p, v, d = element.split('|')
            price.append(float(p))
            volume.append(float(v))
            datetime_int.append(float(d))
    return price, volume, datetime_int

def fit_scalers(csv_file, sample_ratio=1.0):
    df = pd.read_csv(csv_file, header=None).sample(frac=sample_ratio)
    price, volume, datetime_int = [], [], []
    for _, row in df.iterrows():
        p, v, d = series_to_list(row)
        price.extend(p)
        volume.extend(v)
        datetime_int.extend(d)

    return {
        'price': StandardScaler().fit([[p] for p in price]),
        'volume': StandardScaler().fit([[v] for v in volume]),
        'datetime': StandardScaler().fit([[d] for d in datetime_int])
    }

def create_or_load_scalers(processed_dir, update_scalers=False):
    scaler_dir = f'{processed_dir}/scalers'
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_names = ['price', 'volume', 'datetime']
    
    if update_scalers or not all(os.path.exists(os.path.join(scaler_dir, f'{name}_scaler.pkl')) for name in scaler_names):
        scalers = fit_scalers(f'{processed_dir}/train/train.csv', sample_ratio=1.0)
        for name, scaler in scalers.items():
            joblib.dump(scaler, os.path.join(scaler_dir, f'{name}_scaler.pkl'))
        print("Computed and saved new scalers.")
    else:
        scalers = {name: joblib.load(os.path.join(scaler_dir, f'{name}_scaler.pkl')) for name in scaler_names}
        print("Loaded existing scalers.")
    
    return scalers

def estimate_iterations(csv_file, sequence_length, batch_size, sample_ratio=1.0):
    df = pd.read_csv(csv_file, header=None).sample(frac=sample_ratio)
    total_sequences = sum(max(0, len(row.dropna()) - sequence_length + 1) for _, row in df.iterrows())
    estimated_iterations = math.ceil(total_sequences / batch_size)
    print(f"Total sequences: {total_sequences}")
    print(f"Estimated iterations per epoch: {estimated_iterations}")
    return estimated_iterations