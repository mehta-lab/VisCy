from viscy.translation.predict_writer import _pad_shape


def test_pad_shape():
    assert _pad_shape((2, 3), 3) == (1, 2, 3)
    assert _pad_shape((4, 5), 4) == (1, 1, 4, 5)
    full_shape = tuple(range(1, 6))
    assert _pad_shape(full_shape, 5) == full_shape
