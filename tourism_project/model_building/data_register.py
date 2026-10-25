import os
import pandas as pd

def main():
    csv_path = 'tourism_project/data/tourism.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
    
    df = pd.read_csv(csv_path)
    print("Dataset successfully registered!")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    print("Columns:", list(df.columns))
    print(df.info())

if __name__ == '__main__':
    main()
