import argparse
import os
import joblib
import pandas as pd

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
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train', help="Execution mode: 'train' or 'predict'.")
    parser.add_argument('--save_dir', type=str, default='models', help="Directory to save/load models.")
    
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    model_filepath = os.path.join(args.save_dir, f"{args.model}_{args.target_commodity}_model.pkl")

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

    if args.mode == 'train':
        if args.model == 'arima':
            print("\n--- Training ARIMA Model ---")
            trained_model, _, rmse = train_arima_model(target_series, order=tuple(args.arima_order))
            print(f"Final {args.target_commodity} ARIMA RMSE: {rmse:.4f}")

        elif args.model == 'lstm':
            print("\n--- Training LSTM Model ---")
            trained_model, _, rmse = train_lstm_model(target_series, epochs=args.epochs)
            print(f"Final {args.target_commodity} LSTM RMSE: {rmse:.4f}")

        if trained_model:
            joblib.dump(trained_model, model_filepath)
            print(f"Model saved to {model_filepath}")

    elif args.mode == 'predict':
        if not os.path.exists(model_filepath):
            print(f"Error: Trained model not found at {model_filepath}. Please train the model first.")
            return
        
        print(f"\n--- Loading Model from {model_filepath} ---")
        model = joblib.load(model_filepath)

        print("\n--- Making Predictions ---")
        # For demonstration, predict the next 12 months based on the full series
        # In a real scenario, this would be new, unseen data
        forecast_steps = 12
        if args.model == 'arima':
            # ARIMA predict method is different from LSTM
            predictions = model.predict(start=len(target_series), end=len(target_series) + forecast_steps - 1)
            print(f"Forecast for next {forecast_steps} months:\n{predictions}")
        elif args.model == 'lstm':
            print("LSTM prediction logic needs to be implemented separately, as it usually requires sequence generation.")
        
if __name__ == "__main__":
    main()
