# Stock Price Prediction Project

This project implements a machine learning model to predict stock price movements using historical price and volume data.

## Project Structure

```
project_root/
│
├── data/
│   ├── processed/
│   │   ├── features/
│   │   ├── train/
│   │   └── scalers/
│   └── raw/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── utils/
│   ├── evaluation/
│   └── visualization/
│
├── scripts/
│   └── train.py
│
├── config.py
├── stock_data_processor.py
├── main.py
└── README.md
```

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/path-find-er/stock-price-prediction.git
   cd stock-price-prediction
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Prepare your data:
   - Place raw CSV files in the `data/raw/` directory. Each CSV file should contain:
     - `date`: Date and time (format: "YYYY-MM-DD HH:MM:SS")
     - `price`: Stock price for the time period
     - `volume`: Stock volume for the time period
   - Name CSV files with the stock ticker (e.g., `SPY.csv` for S&P 500 ETF)
   - Process the data:
     ```
     python stock_data_processor.py
     ```

## Usage

1. Configure the model and training parameters in `config.py`.

2. Run the main script:
   ```
   python main.py
   ```

3. View the results in the console output and generated plots.

## Components

- `stock_data_processor.py`: Processes raw stock data and prepares it for training.
- `src/data/dataset.py`: Contains the custom PyTorch dataset for stock data.
- `src/models/minLSTM.py`: Defines the LSTM-based prediction model.
- `src/utils/helpers.py`: Contains utility functions used throughout the project.
- `src/evaluation/metrics.py`: Implements evaluation metrics for the model.
- `src/visualization/plots.py`: Creates visualizations of model performance.
- `scripts/train.py`: Implements the training loop for the model.
- `main.py`: Orchestrates the entire process from data preparation to evaluation.

## Contributing

Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

