import pytest
from covid19_sir.model.base import flip_coin


def test_flip_coin():
    # simple assertion test
    assert type(flip_coin(0.5)) == bool

    # expected exceptions
    with pytest.raises(TypeError):
        flip_coin("50")
