#! /usr/bin/env -S python3 -B

"""Test program for the ActiveHDF5Recorder.

We test recording of scalar and array data of compound (record) type,
and recording of scalar and array data of float type.

In each case, the recorder should add a single dimension to allow multiple entries to be stored.
"""

import time

import numpy as np

from hdf5_recorder import ActiveHDF5Recorder

def main() -> None:
    """Test the ActiveHDF5Recorder."""

    record_dtype = np.dtype([
        ("jan"     , np.float64),
        ("piet"    , np.uint32 ),
        ("joris"   , np.uint32 ),
        ("korneel" , np.bool_  )
    ])

    filename = "test.h5"
    frames_per_second = 25.0

    duration = 5.0  # seconds

    with ActiveHDF5Recorder(filename, flush_interval=2.0) as recorder:

        num_frames = round(duration * frames_per_second)

        for frame_index in range(num_frames):
            # Wait for a bit to get to a somewhat realistic frame rate.
            time.sleep(-time.time() % (1.0 / frames_per_second))

            test_percentage = frame_index / num_frames * 100.0
            print("[{:7.3f} %] frame: {:10d} queue: {:10d}".format(test_percentage, frame_index, recorder._queue.qsize()))

            # Make test record data.

            # Note that the creation of a single record is done by calling np.array on a tuple.
            # This is important, as this will create a 0-dimensional (ie scalar) record instance.
            # Somewhat surprisingly, when passing a list as the first argument, this doesn't work as intended.
            record_scalar = np.array((frame_index, frame_index, frame_index, True), dtype=record_dtype)
            assert record_scalar.shape == ()  # empty tuple

            record_1d_array = np.empty(shape=(10,), dtype=record_dtype)
            record_1d_array[:] = record_scalar
            assert record_1d_array.shape == (10, )

            record_2d_array = np.empty(shape=(10, 10), dtype=record_dtype)
            record_2d_array[:, :] = record_scalar
            assert record_2d_array.shape == (10, 10)

            float_scalar = time.time()
            float_1d_array = np.empty(shape=(10,), dtype=np.float64)
            float_1d_array[:] = float_scalar
            assert float_1d_array.shape == (10, )

            float_2d_array = np.empty(shape=(10, 10), dtype=np.float64)
            float_2d_array[:, :] = float_scalar
            assert float_2d_array.shape == (10, 10)

            # Push them to the recorder.

            recorder.store("record_scalar_dset", record_scalar)
            recorder.store("record_1d_array_dset", record_1d_array)
            recorder.store("record_2d_array_dset", record_2d_array)
            recorder.store("float_scalar_dset", float_scalar)
            recorder.store("float_1d_array_dset", float_1d_array)
            recorder.store("float_2d_array_dset", float_2d_array)


if __name__ == "__main__":
    main()
