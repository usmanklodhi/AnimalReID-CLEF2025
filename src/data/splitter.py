import pandas as pd
from sklearn.model_selection import train_test_split

def split_by_identity(df, identity_col='identity', test_size=0.2, random_state=42):
    identity_counts = df[identity_col].value_counts()
    single_occurrence = identity_counts[identity_counts == 1].index
    single_df = df[df[identity_col].isin(single_occurrence)]
    normal_df = df[~df[identity_col].isin(single_occurrence)]
    train_normal, test_normal = train_test_split(
        normal_df, test_size=test_size, stratify=normal_df[identity_col], random_state=random_state
    )
    train_df = pd.concat([train_normal, single_df]).reset_index(drop=True)
    return train_df, test_normal.reset_index(drop=True)

def save_csv(df, path):
    df.to_csv(path, index=False) 