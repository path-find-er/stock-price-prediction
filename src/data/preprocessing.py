import os
import pandas as pd
import joblib
import math
from sklearn.preprocessing import StandardScaler
import logging


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


def fit_scalers(csv_file: str, sample_ratio: float = 1.0) -> dict[str, StandardScaler]:
    df: pd.DataFrame = pd.read_csv(csv_file, header=None).sample(frac=sample_ratio)
    price: list[float] = []
    volume: list[float] = []
    datetime_int: list[float] = []
    for _, row in df.iterrows():
        p, v, d = series_to_list(row)
        price.extend(p)
        volume.extend(v)
        datetime_int.extend(d)

    return {
        "price": StandardScaler().fit([[p] for p in price]),
        "volume": StandardScaler().fit([[v] for v in volume]),
        "datetime": StandardScaler().fit([[d] for d in datetime_int]),
    }


def create_or_load_scalers(
    processed_dir: str, update_scalers: bool = False
) -> dict[str, StandardScaler]:
    scaler_dir: str = f"{processed_dir}/scalers"
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_names: list[str] = ["price", "volume", "datetime"]

    if update_scalers or not all(
        os.path.exists(os.path.join(scaler_dir, f"{name}_scaler.pkl"))
        for name in scaler_names
    ):
        scalers: dict[str, StandardScaler] = fit_scalers(
            f"{processed_dir}/train/train.csv", sample_ratio=1.0
        )
        for name, scaler in scalers.items():
            joblib.dump(scaler, os.path.join(scaler_dir, f"{name}_scaler.pkl"))
        logging.info("Computed and saved new scalers.")
    else:
        scalers: dict[str, StandardScaler] = {
            name: joblib.load(os.path.join(scaler_dir, f"{name}_scaler.pkl"))
            for name in scaler_names
        }
        logging.info("Loaded existing scalers.")

    return scalers


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
