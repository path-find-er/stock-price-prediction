#!/usr/bin/env python3

import os
import random
import argparse
import pandas as pd
from tqdm import tqdm

def process_csv_files(in_folder, out_folder, frac=0.05, random_files=0):
    """
    Process CSV files containing stock data and create train/validation datasets.
    
    Args:
    in_folder (str): Input folder containing CSV files
    out_folder (str): Output folder for processed files
    frac (float): Fraction of data to sample (default: 0.05)
    random_files (int): Number of random files to process in addition to required files (default: 0)
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(out_folder, exist_ok=True)

    # Get list of all CSV files in the input folder
    csv_files = [f for f in os.listdir(in_folder) if f.endswith('.csv')]

    # Sort the csv files by date (assuming filenames are date-based)
    csv_files.sort()

    # Print the selected files
    print("Processing the following files:", csv_files)

    # Initialize empty DataFrames for train and validation data
    train_data = pd.DataFrame()
    val_data = pd.DataFrame()

    # Process each CSV file with a progress bar
    with tqdm(csv_files, desc="Processing CSV files", unit="file") as pbar:
        for csv_file in pbar:
            pbar.set_description(f"Processing {csv_file[:-4]}")
            
            # Load the CSV file into a pandas DataFrame
            df = pd.read_csv(os.path.join(in_folder, csv_file))
            
            # Convert 'Date' column to datetime format
            df['Date'] = pd.to_datetime(df['Date'])

            # Create a DateTimeInt column (Unix timestamp divided by 600 seconds, or 10 minutes)
            df['DateTimeInt'] = (df['Date'].astype(int) // 10**9 // 600).astype(int)
            
            # Extract time and date from the datetime
            df['Time'] = df['Date'].dt.time
            df['Date'] = df['Date'].dt.date

            # Combine price, volume, and DateTimeInt into a single 'data' column
            df['data'] = df['price'].astype(str) + '|' + df['volume'].astype(str) + '|' + df['DateTimeInt'].astype(str)

            # Remove now-redundant columns
            df = df.drop(columns=['price', 'volume', 'DateTimeInt'])

            # Pivot the DataFrame to have Times as columns and Dates as rows
            df_pivot = df.pivot(index='Date', columns='Time', values='data')

            # Reset the index and drop the 'Date' column
            df_pivot.reset_index(inplace=True)
            df_pivot = df_pivot.drop(columns=['Date'])

            # Get the ticker from the filename and rename columns to include the ticker
            ticker = os.path.splitext(csv_file)[0]
            df_pivot.columns = [f"{col}-{ticker}" for col in df_pivot.columns]
            
            # Split into train and validation sets
            train_start = int(len(df_pivot) * 0.05)
            train_end = int(len(df_pivot) * 0.90)
            val_df = pd.concat([df_pivot.iloc[:train_start], df_pivot.iloc[train_end:]])
            train_df = df_pivot.iloc[train_start:train_end]
            
            # Transpose the dataframes
            train_df = train_df.T
            val_df = val_df.T

            # Append to the respective DataFrames
            train_data = pd.concat([train_data, train_df], ignore_index=True)
            val_data = pd.concat([val_data, val_df], ignore_index=True)
            
            pbar.update()

    # Reset indices
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    # Shuffle the rows and sample a fraction of the data
    train_data = train_data.sample(frac=frac).reset_index(drop=True)
    val_data = val_data.sample(frac=frac).reset_index(drop=True)

    # Save the processed DataFrames to new CSV files in the out_folder
    train_data.to_csv(os.path.join(out_folder, 'train.csv'), index=False, header=False)
    val_data.to_csv(os.path.join(out_folder, 'val.csv'), index=False, header=False)

    print(f"Processed data saved to {out_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process stock data CSV files and create train/validation datasets.")
    parser.add_argument("--in_folder", default="processed/features", help="Input folder containing CSV files")
    parser.add_argument("--out_folder", default="processed/train", help="Output folder for processed files")
    parser.add_argument("--frac", type=float, default=0.05, help="Fraction of data to sample (default: 0.05)")
    parser.add_argument("--random_files", type=int, default=0, help="Number of random files to process in addition to required files (default: 0)")
    
    args = parser.parse_args()
    
    process_csv_files(args.in_folder, args.out_folder, args.frac, args.random_files)