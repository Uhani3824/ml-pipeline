from src.preprocess import load_data, preprocess

def test_load_data():
    df = load_data()
    assert not df.empty

def test_preprocess_output_shapes():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
