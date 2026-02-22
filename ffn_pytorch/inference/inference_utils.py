"""FFN inference utilities (PyTorch).

This module provides common utilities for inference:

1. Counter system (Counters + StatCounter):
   - Thread-safe hierarchical counters (replaces TF TFSyncVariable)
   - Increment/Set/IncrementBy
   - Parent-child (child updates propagate to parent)
   - Serialize/deserialize (for checkpoints)

2. Timing (timer_counter):
   - Context manager tracking call count and cumulative time
   - Usage: with timer_counter(counters, 'name'): ...

3. TimedIter:
   - Wraps an iterator with timing

4. Histogram tools (match_histogram, compute_histogram_lut):
   - CLAHE + histogram matching for intensity normalization in preprocessing
"""

import contextlib
import json
import threading
import time

import numpy as np
import skimage.exposure

from . import storage


class StatCounter:
    """Stat counter with a MR counter interface."""

    def __init__(self, update, name, parent=None):
        self._counter = 0
        self._update = update
        self._lock = threading.Lock()
        self._parent = parent

    def Increment(self):
        self.IncrementBy(1)

    def IncrementBy(self, x, export=True):
        with self._lock:
            self._counter += int(x)
            self._update()
        if self._parent is not None:
            self._parent.IncrementBy(x)

    def Set(self, x, export=True):
        x_old = self._counter
        x_diff = x - x_old
        self.IncrementBy(x_diff, export=export)

    def __repr__(self):
        return 'StatCounter(total=%g)' % (self.value)

    @property
    def value(self):
        return self._counter


MSEC_IN_SEC = 1000


class Counters:
    """Container for counters."""

    def __init__(self, parent=None):
        self._lock = threading.Lock()
        self.reset()
        self.parent = parent

    def reset(self):
        with self._lock:
            self._counters = {}
        self._last_update = 0

    def __getitem__(self, name: str) -> StatCounter:
        return self.get(name)

    def get(self, name: str, **kwargs) -> StatCounter:
        with self._lock:
            if name not in self._counters:
                self._counters[name] = self._make_counter(name, **kwargs)
            return self._counters[name]

    def __iter__(self):
        return iter(self._counters.items())

    def _make_counter(self, name: str, **kwargs) -> StatCounter:
        del kwargs
        return StatCounter(self.update_status, name)

    def update_status(self):
        pass

    def get_sub_counters(self):
        return Counters(self)

    def dump(self, filename: str):
        with storage.atomic_file(filename, 'w') as fd:
            for name, counter in sorted(self._counters.items()):
                fd.write('%s: %d\n' % (name, counter.value))

    def dumps(self) -> str:
        state = {name: counter.value for name, counter in self._counters.items()}
        return json.dumps(state)

    def loads(self, encoded_state: str):
        state = json.loads(encoded_state)
        for name, value in state.items():
            self[name].Set(value, export=False)


@contextlib.contextmanager
def timer_counter(counters: Counters, name: str, export=True, increment=1):
    """Creates a counter tracking time spent in the context."""
    assert isinstance(counters, Counters)
    counter = counters.get(name + '-calls', export=export)
    timer = counters.get(name + '-time-ms', export=export)
    start_time = time.time()
    try:
        yield timer, counter
    finally:
        counter.IncrementBy(increment)
        dt = (time.time() - start_time) * MSEC_IN_SEC
        timer.IncrementBy(dt)


class TimedIter:
    """Wraps an iterator with a timing counter."""

    def __init__(self, it, counters, counter_name):
        self.it = it
        self.counters = counters
        self.counter_name = counter_name

    def __iter__(self):
        return self

    def __next__(self):
        with timer_counter(self.counters, self.counter_name):
            ret = next(self.it)
        return ret

    def next(self):
        return self.__next__()


def match_histogram(image, lut, mask=None):
    """Changes the intensity distribution of a 3d image."""
    for z in range(image.shape[0]):
        clahe_slice = skimage.exposure.equalize_adapthist(image[z, ...])
        clahe_slice = (clahe_slice * 255).astype(np.uint8)

        valid_slice = clahe_slice
        if mask is not None:
            valid_slice = valid_slice[np.logical_not(mask[z, ...])]

        if valid_slice.size == 0:
            continue

        cdf, bins = skimage.exposure.cumulative_distribution(valid_slice)
        cdf = np.array(cdf.tolist() + [1.0])
        bins = np.array(bins.tolist() + [255])
        image[z, ...] = lut[
            (cdf[np.searchsorted(bins, clahe_slice)] * 255).astype(np.uint8)]


def compute_histogram_lut(image):
    """Computes the inverted CDF of image intensity."""
    cdf, bins = skimage.exposure.cumulative_distribution(image)
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(0, 256):
        lut[i] = bins[np.searchsorted(cdf, i / 255.0)]
    return lut
