# Commodity Price Prediction

This project provides a command-line tool for predicting commodity prices (e.g., Gold, Silver) using time-series machine learning models like ARIMA and LSTM. It integrates various economic indicators as features for improved prediction accuracy.

The original analysis was performed in a Wolfram Mathematica notebook and has been ported and refactored into a structured Python application.

## Project Overview

The tool executes the following pipeline:
1.  **Data Loading & Cleaning:** Reads various CSV files (commodity prices, inflation, interest rates, monetary base, etc.), cleans them, and merges them into a unified, date-indexed dataset.
2.  **Time-Series Preprocessing:** Resamples the data to a monthly frequency and handles missing values.
3.  **Model Training:** Supports training of ARIMA models (and a conceptual LSTM model) on historical price data.
4.  **Model Persistence:** Saves trained models for later use.
5.  **Price Prediction:** Generates future price forecasts using loaded models.

## Project Structure

-   `data/`: Contains all raw input CSV data files.
-   `src/data_processing.py`: Handles loading, cleaning, and merging of raw data.
-   `src/time_series_utils.py`: Provides utility functions for time-series manipulation (e.g., resampling).
-   `src/models.py`: Defines the time-series models (ARIMA implementation, LSTM placeholder) and their training/evaluation logic.
-   `main.py`: The main command-line interface (CLI) script to orchestrate the entire workflow.
-   `requirements.txt`: Lists all necessary Python dependencies.

---

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/adityasengar/commodity_price_prediction.git
    cd commodity_price_prediction
    ```

2.  It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

The `main.py` script is used to train models and make predictions. It operates in two modes: `train` and `predict`.

### Training a Model

To train an ARIMA model for Gold prices:

```bash
python main.py --mode train --target_commodity gold --model arima --arima_order 5 1 0
```

This will:
-   Load data from the `data/` directory.
-   Train an ARIMA(5,1,0) model on the Gold price series.
-   Save the trained model to `models/arima_gold_model.pkl`.

### Making Predictions

To predict future Gold prices using a trained ARIMA model:

```bash
python main.py --mode predict --target_commodity gold --model arima
```

This will:
-   Load the trained ARIMA model for Gold from `models/arima_gold_model.pkl`.
-   Generate a forecast for the next 12 months.

### Command-Line Arguments

-   `--data_dir`: Directory containing the CSV data files. (Default: `.`).
-   `--target_commodity`: The commodity to predict (`gold`, `silver`, etc.). (Default: `gold`).
-   `--model`: The time-series model to use (`arima` or `lstm`). (Default: `arima`).
-   `--epochs`: Number of epochs for LSTM training (if applicable). (Default: `100`).
-   `--arima_order`: ARIMA model order as three integers (p, d, q). (Default: `5 1 0`).
-   `--mode`: `train` or `predict`. (Default: `train`).
-   `--save_dir`: Directory to save/load models. (Default: `models`).

# Updated on 2026-01-09
