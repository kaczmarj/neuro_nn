import multiprocessing
import os
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from nobrainer.io import read_csv
import nobrainer.utils as nbutils


def test_get_data():
    csv_path = nbutils.get_data()
    assert Path(csv_path).is_file()

    files = read_csv(csv_path)
    assert len(files) == 10
    assert all(len(r) == 2 for r in files)
    for x, y in files:
        assert Path(x).is_file()
        assert Path(y).is_file()


def test_get_all_cpus():
    assert nbutils._get_all_cpus() == multiprocessing.cpu_count()
    os.environ['SLURM_CPUS_ON_NODE'] = "128"
    assert nbutils._get_all_cpus() == 128


def test_streaming_stats():
    # TODO: add entropy
    ss = nbutils.StreamingStats()
    xs = np.random.random_sample((100))
    for x in xs:
        ss.update(x)
    assert_allclose(xs.mean(), ss.mean())
    assert_allclose(xs.std(), ss.std())
    assert_allclose(xs.var(), ss.var())

    ss = nbutils.StreamingStats()
    xs = np.random.random_sample((10, 5, 5, 5))
    for x in xs:
        ss.update(x)
    assert_allclose(xs.mean(0), ss.mean())
    assert_allclose(xs.std(0), ss.std())
    assert_allclose(xs.var(0), ss.var())
