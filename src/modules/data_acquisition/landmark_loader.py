import pandas as pd


def load_landmarks(txt_file_path):
    """
    Reads a comma-separated .txt file and extracts facial landmark coordinates.

    Args:
    - txt_file_path (str): Path to the .txt file containing landmark data.

    Returns:
    - DataFrame: A pandas DataFrame with columns for frame number and landmark coordinates.
    """
    try:
        # Read the text file as a CSV with the first row as the header
        df = pd.read_csv(txt_file_path, delimiter=',')

        # Verify that the required columns are present
        if 'Frame' not in df.columns or len(df.columns) < 137:
            print("File format error: Ensure the file has 'Frame' and 136 landmark columns.")
            return None

        return df
    except Exception as e:
        print(f"Error loading landmarks from {txt_file_path}: {e}")
        return None
