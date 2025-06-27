from src.utils.early_stopping import EarlyStopping

def test_early_stopping():
    es = EarlyStopping(patience=2, min_delta=0.1)
    losses = [1.0, 0.95, 0.96, 0.97, 0.98]
    stops = []
    for loss in losses:
        stops.append(es.should_stop(loss))
    # Should not stop until patience is exceeded
    assert stops[-1] == True 