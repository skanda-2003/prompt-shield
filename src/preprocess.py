import pandas as pd
import os
from sklearn.model_selection import train_test_split

def clean_text(df):
    # strip leading/trailing whitespace from every text entry
    df['text'] = df['text'].str.strip()
    # drop rows where text is empty after stripping
    df = df[df['text'] != '']
    return df

def split_data(df):
    train_df, temp = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)
    return train_df, val_df, test_df


if __name__ == '__main__':
    df = pd.read_csv('data/processed/combined.csv')

    df = clean_text(df)
    train, val, test = split_data(df)

    # save each split - index=False prevents pandas writing a row number column
    train.to_csv('data/processed/train.csv', index=False)
    val.to_csv('data/processed/val.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)

    print(f'Train: {len(train)} rows')
    print(f'Val:   {len(val)} rows')
    print(f'Test:  {len(test)} rows')