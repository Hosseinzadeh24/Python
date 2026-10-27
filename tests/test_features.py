import numpy as np

from crypto_bot.data import synthetic_ohlcv
from crypto_bot.features import build_features, forward_labels


def test_features_do_not_use_future_data():
    df = synthetic_ohlcv(n_bars=400, seed=3)
    f_full = build_features(df)
    # change everything AFTER bar 300; features up to bar 300 must not change
    df2 = df.copy()
    df2.iloc[301:] = df2.iloc[301:] * 3.0
    f_mod = build_features(df2)
    a = f_full.iloc[:301].to_numpy()
    b = f_mod.iloc[:301].to_numpy()
    assert np.allclose(a, b, equal_nan=True)


def test_labels_use_future_and_are_nan_at_end():
    df = synthetic_ohlcv(n_bars=50, seed=1)
    fwd, lab = forward_labels(df, horizon=4, cost=0.001)
    assert fwd.iloc[-4:].isna().all()
    assert lab.iloc[-4:].isna().all()
    assert set(lab.dropna().unique()) <= {0.0, 1.0}


def test_feature_matrix_has_expected_columns():
    f = build_features(synthetic_ohlcv(n_bars=200, seed=2))
    for col in ("rsi_14", "macd_hist", "bb_z", "vol_z", "sma10_sma30", "atr_pct"):
        assert col in f.columns
