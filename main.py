import argparse
import os

from src.data_processing import load_and_clean_data
from src.time_series_utils import monthly_resample
from src.models import train_arima_model, train_lstm_model

def main():
    """Main function to run the commodity price prediction workflow."""
    parser = argparse.ArgumentParser(description="Commodity Price Prediction")
    parser.add_argument('--data_dir', type=str, default='.', help="Directory containing the CSV data files.")
    parser.add_argument('--target_commodity', type=str, default='gold', help="The commodity to predict (e.g., 'gold', 'silver').")
    parser.add_argument('--model', type=str, choices=['arima', 'lstm'], default='arima', help="Model to use for prediction.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for LSTM training (if applicable).")
    parser.add_argument('--arima_order', type=int, nargs=3, default=[5,1,0], help="ARIMA model order (p,d,q).")
    
    args = parser.parse_args()

    print("--- Commodity Price Prediction Pipeline ---")
    
    # 1. Load and clean data
    df = load_and_clean_data(data_dir=args.data_dir)
    if df is None:
        print("Exiting due to data loading errors.")
        return
    
    # 2. Resample to monthly frequency
    df_monthly = monthly_resample(df)
    
    # Select target series
    if args.target_commodity not in df_monthly.columns:
        print(f"Error: Target commodity '{args.target_commodity}' not found in data. Available: {df_monthly.columns.tolist()}")
        return
    
    target_series = df_monthly[args.target_commodity]
    print(f"\nSelected target: {args.target_commodity} series shape: {target_series.shape}")

    # 3. Model Training
    if args.model == 'arima':
        print("\n--- Training ARIMA Model ---")
        model, predictions, rmse = train_arima_model(target_series, order=tuple(args.arima_order))
        print(f"Final {args.target_commodity} ARIMA RMSE: {rmse:.4f}")

    elif args.model == 'lstm':
        print("\n--- Training LSTM Model ---")
        model, predictions, rmse = train_lstm_model(target_series, epochs=args.epochs)
        print(f"Final {args.target_commodity} LSTM RMSE: {rmse:.4f}")

    # In the next step, we will implement model saving and actual prediction.

if __name__ == "__main__":
    main()