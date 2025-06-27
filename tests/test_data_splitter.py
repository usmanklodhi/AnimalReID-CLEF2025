import pandas as pd
from src.data import splitter

def test_split_by_identity():
    df = pd.DataFrame({
        'identity': ['a', 'a', 'b', 'b', 'c'],
        'val': [1, 2, 3, 4, 5]
    })
    train, test = splitter.split_by_identity(df, identity_col='identity', test_size=0.5, random_state=0)
    # 'c' only occurs once, should be in train
    assert 'c' in train['identity'].values
    assert 'c' not in test['identity'].values
    # All identities in test should be in train
    assert set(test['identity']).issubset(set(train['identity']))

def test_save_csv(tmp_path):
    df = pd.DataFrame({'a': [1, 2]})
    out_path = tmp_path / 'out.csv'
    splitter.save_csv(df, str(out_path))
    loaded = pd.read_csv(out_path)
    assert loaded.equals(df) 