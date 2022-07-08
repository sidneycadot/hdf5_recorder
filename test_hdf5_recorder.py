#! /usr/bin/env -S python3 -B

"""Test program for the ActiveHDF5Recorder."""

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

    filename = "test_directory/test.h5"
    frames_per_second = 25.0

    with ActiveHDF5Recorder(filename, flush_interval=2.0) as recorder:

        num_frames = round(300.0 * frames_per_second)  # Test for 5 minutes

        for frame_index in range(num_frames):
            # Wait for a bit to get to a somewhat realistic frame rate.
            time.sleep(-time.time() % (1.0 / frames_per_second))

            test_percentage = frame_index / num_frames * 100.0
            print("[{:7.3f} %] frame: {:10d} queue: {:10d}".format(test_percentage, frame_index, recorder._queue.qsize()))

            # Make test frame data.
            frame = np.zeros((2500, 2500), dtype=np.uint16) + frame_index
            frame_time = time.time()

            # Make test record data.
            # Note that the creation of a single record is done by calling np.array on a tuple.
            # This is important, as this will create a 0-dimensional (ie scalar) record instance, as intended;
            # Somewhat surprisingly, when passing a list as the first argument, this doesn't work as intended.
            record = np.array((frame_index, frame_index, frame_index, True), dtype=record_dtype)

            # Verify that the record is, in fact, 0-dimensional as intended.
            assert len(record.shape) == 0
            record_time = time.time()

            # Push them to the recorder.
            recorder.store("frames", frame)
            recorder.store("frame_time", frame_time)
            recorder.store("records", record)
            recorder.store("record_time", record_time)


if __name__ == "__main__":
    main()
