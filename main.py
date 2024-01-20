import os
from src.data_processing import load_and_clean_data
from src.time_series_utils import monthly_resample

def main():
    """Main function to run the commodity price prediction workflow."""
    print("--- Commodity Price Prediction Pipeline ---")
    
    # 1. Load and clean data
    df = load_and_clean_data(data_dir=".")
    if df is None:
        print("Exiting due to data loading errors.")
        return
    
    # 2. Resample to monthly frequency
    df_monthly = monthly_resample(df)
    
    print("\n--- Processed Monthly Data Head ---")
    print(df_monthly.head())
    print("\n--- Processed Monthly Data Info ---")
    df_monthly.info()
    
    print("\nData processing complete. Next steps would involve modeling.")

if __name__ == "__main__":
    main()

