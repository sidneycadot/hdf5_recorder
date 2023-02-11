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

    with ActiveHDF5Recorder(filename, flush_interval=2.0) as recorder:

        num_loops = 10

        for loop_index in range(num_loops):

            # Make test record data.

            # Note that the creation of a single record is done by calling np.array on a tuple.
            # This is important, as this will create a 0-dimensional (ie scalar) record instance.
            # Somewhat surprisingly, when passing a list as the first argument, this doesn't work as intended.
            record_scalar = np.array((loop_index, loop_index, loop_index, True), dtype=record_dtype)
            assert record_scalar.shape == ()  # empty tuple

            record_1d_array = np.empty(shape=(17,), dtype=record_dtype)
            record_1d_array[:] = record_scalar
            assert record_1d_array.shape == (17, )

            record_2d_array = np.empty(shape=(23, 29), dtype=record_dtype)
            record_2d_array[:, :] = record_scalar
            assert record_2d_array.shape == (23, 29)

            float_scalar = loop_index
            float_1d_array = np.empty(shape=(17,), dtype=np.float64)
            float_1d_array[:] = float_scalar
            assert float_1d_array.shape == (17, )

            float_2d_array = np.empty(shape=(23, 29), dtype=np.float64)
            float_2d_array[:, :] = float_scalar
            assert float_2d_array.shape == (23, 29)

            # Push them to the recorder in 'append' mode:

            recorder.append("append_record_scalar_dset", record_scalar)
            recorder.append("append_record_1d_array_dset", record_1d_array)
            recorder.append("append_record_2d_array_dset", record_2d_array)

            recorder.append("append_float_scalar_dset", float_scalar)
            recorder.append("append_float_1d_array_dset", float_1d_array)
            recorder.append("append_float_2d_array_dset", float_2d_array)

            # We cannot "extend" with scalars!

            #recorder.extend("extend_record_scalar_dset", record_scalar)
            recorder.extend("extend_record_1d_array_dset", record_1d_array)
            recorder.extend("extend_record_2d_array_dset", record_2d_array)

            #recorder.extend("extend_float_scalar_dset", float_scalar)
            recorder.extend("extend_float_1d_array_dset", float_1d_array)
            recorder.extend("extend_float_2d_array_dset", float_2d_array)

    print("All done.")


if __name__ == "__main__":
    main()
