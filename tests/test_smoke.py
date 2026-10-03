from swissdelay import __version__, config


def test_version():
    assert __version__ == "0.1.0"


def test_splits_are_ordered():
    assert (
        config.PERIOD_START
        < config.TRAIN_END
        < config.VALID_END
        < config.TEST_END
        < config.PERIOD_END
    )


def test_horizons_sorted():
    assert list(config.HORIZONS_MIN) == sorted(config.HORIZONS_MIN)
