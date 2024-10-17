from viscy.translation.predict_writer import _pad_shape


def test_pad_shape():
    assert _pad_shape((2, 3), 3) == (1, 2, 3)
    assert _pad_shape((4, 5), 4) == (1, 1, 4, 5)
    assert _pad_shape((2, 3, 4, 5), 5) == (1, 2, 3, 4, 5)
