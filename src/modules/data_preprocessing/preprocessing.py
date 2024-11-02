import pandas as pd

def normalize_landmarks(landmarks_df):
    normalized_df = landmarks_df.copy()
    for i in range(1, 69):
        x_col, y_col = f'x{i}', f'y{i}'
        if x_col in normalized_df.columns and y_col in normalized_df.columns:
            normalized_df[x_col] = (normalized_df[x_col] - normalized_df[x_col].mean()) / normalized_df[x_col].std()
            normalized_df[y_col] = (normalized_df[y_col] - normalized_df[y_col].mean()) / normalized_df[y_col].std()
    return normalized_df
