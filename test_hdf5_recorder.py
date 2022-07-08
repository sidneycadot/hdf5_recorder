#! /usr/bin/env -S python3 -B -u

"""Test program for the ActiveHDF5Recorder."""

import time

import numpy as np

from hdf5_recorder import ActiveHDF5Recorder

def main() -> None:

    # Test the ActiveHDF5Recorder.

    imu_dtype = np.dtype([
        ("jan"     , np.float64),
        ("piet"    , np.uint32 ),
        ("joris"   , np.uint32 ),
        ("korneel" , np.bool_  )
    ])

    filename = "/home/sidney/vm_shared/test.h5"
    FPS = 25

    with ActiveHDF5Recorder(filename, flush_interval=2.0) as recorder:

        num_frames = 3600 * FPS

        for frame_index in range(num_frames):
            # Wait for a bit to get to a somewhat realistic frame rate.
            time.sleep(-time.time() % (1.0 / FPS))
            
            test_percentage = frame_index / num_frames * 100.0
            print("[{:7.3f} %] frame: {:10d} queue: {:10d}".format(test_percentage, frame_index, recorder._queue.qsize()))

            # Make test frame and test IMU data.
            frame = np.zeros((2500, 2500), dtype=np.uint16) + frame_index
            frame_time = time.time()
            imu_record = np.array((frame_index, frame_index, frame_index, True), dtype=imu_dtype)
            imu_record_time = time.time()

            # Push them to the recorder.
            recorder.store("frames", frame)
            recorder.store("frame_time", frame_time)
            recorder.store("imu_records", imu_record)
            recorder.store("imu_record_time", imu_record_time)

if __name__ == "__main__":
    main()
