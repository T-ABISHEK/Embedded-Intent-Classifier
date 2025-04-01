import pandas as pd

def load_csv(file_path: str):
    df = pd.read_csv(file_path)
    if 'prompt' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'prompt' and 'label' columns.")
    return df['prompt'].tolist(), df['label'].tolist()