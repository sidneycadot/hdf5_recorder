"""This module provides the HDF5Recorder and ActiveHDF5Recorder classes."""

import queue
import time
import multiprocessing
from typing import Optional, Dict, List

import numpy as np
import h5py


class HDF5Recorder:
    """This class provides HDF5 file output, buffering data in memory until flush() is called.

    This class is intended for simple, single-threaded use cases where the fact that flush() may take
    considerable time is not an issue.

    For more demanding use-cases, consider using the ActiveHDF5Recorder class defined below.

    Note that the HDF5Recorder does not normally keep the HDF5 file open; it is re-opened every time
    data needs to be written inside the flush() routine. This is done on purpose. In long-running
    processes, it can be useful to have a look at partially written data, and an open HDF5 file cannot
    be opened by a second process. By having the HDF5 file closed most of the time, we can make a
    valid copy of it that can be used for inspection and analysis.

    Mode "extend": the stored element is a (possibly multi-dimensional) array. The content will be
    appended to the pre-existing dataset. The type of the dataset and the element to be stored
    should be identical, EXCEPT for the first dimension.

    Mode 'stack": the stored element is assumed to be a single element that will be stacked on
      (after) any pre-existing elements found in the dataset.

      The dataset will have an extra leading dimension compared to the elements.

      All elements should have the same shape and dtype.
    """

    def __init__(self, filename: str):
        self._filename = filename
        self._store_data: Dict[str, List[np.ndarray]] = {}
        self._is_open = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()

    def open(self) -> None:
        """Open the HDF5 recorder."""
        if self._is_open:
            raise RuntimeError("Attempt to open an HDF5Recorder that is already open.")

        # Create file, truncate if it already exists.
        with h5py.File(self._filename, "w"):
            # Create an empty HDF5 file and close it immediately.
            pass

        self._is_open = True

    def close(self) -> None:
        """Close the HDF5 recorder."""
        if not self._is_open:
            raise RuntimeError("Attempt to close an HDF5Recorder that is already closed.")
        self.flush()
        self._is_open = False

    def store(self, dataset: str, data: np.ndarray, mode: str='extend') -> None:
        """Buffer data inside the HDF5 recorder, intended to be stored at the next invocation of flush()."""
        if not self._is_open:
            raise RuntimeError("Attempt to store data to an HDF5Recorder that is closed.")

        if dataset not in self._store_data:
            self._store_data[dataset] = [data]
        else:
            self._store_data[dataset].append(data)

    def flush(self) -> None:
        """Flush data from the HDF5 recorder to file.

        This method may take a considerable amount of time (seconds or more) if a considerable amount of data
        is currently buffered.
        """

        if not self._is_open:
            raise RuntimeError("Attempt to flush an HDF5Recorder that is closed.")

        if len(self._store_data) == 0:
            return

        # Open the HDF5 file for read/write; the file must exist.
        with h5py.File(self._filename, "r+") as hdf5_file:

            for (key, data_items) in self._store_data.items():

                # Combine all data items.
                # This adds an extra dimension in front, even if it's just a single item.
                data_array = np.stack(data_items)

                if key in hdf5_file:
                    # The dataset already exists.
                    # Reserve space for data in the dataset, and put it there.
                    dset = hdf5_file[key]
                    old_size = len(dset)
                    new_size = old_size + len(data_array)
                    # print("enlarge dataset:", key, old_size, new_size)
                    dset.resize(new_size, axis=0)
                    dset[old_size:new_size]=data_array
                else:
                    # Create the dataset with the first (outer) dimension resizable.
                    # print("create dataset:", key, len(data_array))
                    hdf5_file.create_dataset(key, data=data_array, maxshape=(None, ) + data_array.shape[1:])

        # Clear the buffer.
        self._store_data.clear()


def _active_hdf5_recorder(filename: str, flush_interval: float, the_queue: multiprocessing.Queue) -> None:
    """This function creates an HDF5Recorder instance and feeds data from the queue to it.

    In addition, it calls the HDF5Recorder's flush() method, 'flush_interval' seconds after
    completion of the previous flush().

    The function ends when an end-of-stream sentinel element (None) is received via the queue.
    """

    with HDF5Recorder(filename) as recorder:

        flush_time = time.monotonic() + flush_interval

        while True:

            max_block_time = max(0.0, flush_time - time.monotonic())

            try:
                item = the_queue.get(timeout=max_block_time)
                if item is None:
                    # Detected end-of-stream sentinel.
                    break
            except queue.Empty:
                # Time-out on get(); no data queued for recording.
                item = None

            if item is not None:
                (dataset, data) = item
                recorder.store(dataset, data)

            if time.monotonic() >= flush_time:
                recorder.flush()
                flush_time = time.monotonic() + flush_interval


class ActiveHDF5Recorder:
    """This class provides HDF5 file output, flushing data to file at a configurable interval.

    The flush() operation does not have to be (in fact, it cannot) be called explicitly.

    The ActiveHDF5Recorder uses a sub-process to do the actual HDF5 writing. The advantage
    of this is that the slow flush() operation will not impact the thread that uses the
    ActiveHDF5Recorder.
    """

    def __init__(self, filename: str, flush_interval: float):
        self._filename = filename
        self._flush_interval = flush_interval
        self._queue: Optional[multiprocessing.Queue] = None
        self._process: Optional[multiprocessing.Process] = None
        self._is_open = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()

    def open(self) -> None:
        """Open the ActiveHDF5Recorder."""

        if self._is_open:
            raise RuntimeError("Attempt to open an ActiveHDF5Recorder that is already open.")

        self._queue = multiprocessing.Queue()
        self._process = multiprocessing.Process(target=_active_hdf5_recorder, args=(self._filename, self._flush_interval, self._queue))
        self._process.start()
        self._is_open = True

    def close(self) -> None:
        """Close the ActiveHDF5Recorder."""

        if not self._is_open:
            raise RuntimeError("Attempt to close an ActiveHDF5Recorder that is already closed.")

        # These asserts will always pass. We added them as analysis hints for mypy.
        assert self._queue is not None
        assert self._process is not None

        # Insert end-of-stream sentinel into the stream.
        self._queue.put(None)

        # Wait until the process has terminated, then free its resources.
        self._process.join()

        self._queue = None
        self._process = None

        self._is_open = False

    def store(self, dataset: str, data: np.ndarray, mode: str='extend') -> None:
        """Store data into the ActiveHDF5Recorder."""

        if not self._is_open:
            raise RuntimeError("Attempt to store data to an ActiveHDF5Recorder that is closed.")

        # This assert will always pass. We added it as an analysis hint for mypy.
        assert self._queue is not None

        self._queue.put((dataset, data))
