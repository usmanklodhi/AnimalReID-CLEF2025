import pandas as pd
import os
from src.data import loader

def test_load_metadata(tmp_path):
    # Create a dummy CSV
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_csv(csv_path, index=False)
    loaded = loader.load_metadata(str(csv_path))
    assert loaded.equals(df)

def test_show_info(capsys):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    loader.show_info(df)
    captured = capsys.readouterr()
    assert "a" in captured.out and "b" in captured.out 