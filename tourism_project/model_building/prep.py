import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    csv_path = 'tourism_project/data/tourism.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
    
    df = pd.read_csv(csv_path)
    
    # Drop uniquely identifying columns if they exist
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
        
    # Split data into Train and Test sets (e.g., 80-20 split)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save local CSV splits
    train_df.to_csv('tourism_project/model_building/train.csv', index=False)
    test_df.to_csv('tourism_project/model_building/test.csv', index=False)
    print("Data preprocessed and split into train.csv and test.csv")

if __name__ == '__main__':
    main()
