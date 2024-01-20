import pandas as pd
import os

def load_and_clean_data(data_dir="."):
    """
    Loads all relevant CSV files, cleans them, and merges them into a single
    pandas DataFrame indexed by date.
    
    Args:
        data_dir (str): The directory where the CSV files are located.

    Returns:
        pd.DataFrame: A cleaned and merged DataFrame.
    """
    print("Loading and cleaning data...")
    
    # Define files to load
    files_to_load = {
        "gold": "gold_price.csv",
        "silver": "silver_price.csv",
        "cpi": "CPI.csv",
        "interest": "interest_rate.csv",
        "monetary_base": "monetary_base.csv",
        "snp": "snp_index.csv",
        "gsci": "gsci.csv",
        "yield": "Treasury_yield.csv",
    }

    data_frames = {}
    
    for name, filename in files_to_load.items():
        try:
            path = os.path.join(data_dir, filename)
            # Assuming CSVs have 'Date' and value columns
            df = pd.read_csv(path, parse_dates=['Date'])
            df = df.set_index('Date')
            # A more robust implementation would handle column name differences
            data_frames[name] = df.iloc[:, 0] # Assume value is in the first column
        except Exception as e:
            print(f"Warning: Could not load or process {filename}. Error: {e}")
    
    # Combine all series into a single DataFrame
    full_df = pd.DataFrame(data_frames)
    full_df = full_df.sort_index()
    
    # Forward-fill missing values, a common strategy for time series
    full_df = full_df.ffill()
    full_df = full_df.dropna()
    
    print(f"Data loading complete. DataFrame shape: {full_df.shape}")
    return full_df

if __name__ == '__main__':
    df = load_and_clean_data()
    print("\n--- Data Head ---")
    print(df.head())
    print("\n--- Data Info ---")
    df.info()
