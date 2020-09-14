import h5netcdf_

ENGINES = {
    "h5netcdf": h5netcdf_.open_v2
}


def open_dataset(
    filename_or_obj,
    group=None,
    decode_cf=True,
    mask_and_scale=None,
    decode_times=True,
    concat_characters=True,
    decode_coords=True,
    engine=None,
    chunks=None,
    lock=None,
    cache=None,
    drop_variables=None,
    backend_kwargs=None,
    use_cftime=None,
    decode_timedelta=None,
):
    pass