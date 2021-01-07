import os
from io import BytesIO
from typing import Mapping
from pathlib import Path

from .common import ArrayWriter
from ..core import indexing
from ..core.dataset import Dataset, _get_chunk, _maybe_chunk
from ..core.utils import is_remote_uri
from . import plugins


def _protect_dataset_variables_inplace(dataset, cache):
    for name, variable in dataset.variables.items():
        if name not in variable.dims:
            # no need to protect IndexVariable objects
            data = indexing.CopyOnWriteArray(variable._data)
            if cache:
                data = indexing.MemoryCachedArray(data)
            variable.data = data


def _get_mtime(filename_or_obj):
    # if passed an actual file path, augment the token with
    # the file modification time
    mtime = None

    try:
        path = os.fspath(filename_or_obj)
    except TypeError:
        path = None

    if path and not is_remote_uri(path):
        mtime = os.path.getmtime(filename_or_obj)

    return mtime


def _chunk_ds(
    backend_ds,
    filename_or_obj,
    engine,
    chunks,
    overwrite_encoded_chunks,
    **extra_tokens,
):
    from dask.base import tokenize

    mtime = _get_mtime(filename_or_obj)
    token = tokenize(filename_or_obj, mtime, engine, chunks, **extra_tokens)
    name_prefix = "open_dataset-%s" % token

    variables = {}
    for name, var in backend_ds.variables.items():
        var_chunks = _get_chunk(var, chunks)
        variables[name] = _maybe_chunk(
            name,
            var,
            var_chunks,
            overwrite_encoded_chunks=overwrite_encoded_chunks,
            name_prefix=name_prefix,
            token=token,
        )
    ds = backend_ds._replace(variables)
    return ds


def _dataset_from_backend_dataset(
    backend_ds,
    filename_or_obj,
    engine,
    chunks,
    cache,
    overwrite_encoded_chunks,
    **extra_tokens,
):
    if not (isinstance(chunks, (int, dict)) or chunks is None):
        if chunks != "auto":
            raise ValueError(
                "chunks must be an int, dict, 'auto', or None. "
                "Instead found %s. " % chunks
            )

    _protect_dataset_variables_inplace(backend_ds, cache)
    if chunks is None:
        ds = backend_ds
    else:
        ds = _chunk_ds(
            backend_ds,
            filename_or_obj,
            engine,
            chunks,
            overwrite_encoded_chunks,
            **extra_tokens,
        )

    ds._file_obj = backend_ds._file_obj

    # Ensure source filename always stored in dataset object (GH issue #2550)
    if "source" not in ds.encoding:
        if isinstance(filename_or_obj, str):
            ds.encoding["source"] = filename_or_obj

    return ds


def _resolve_decoders_kwargs(decode_cf, open_backend_dataset_parameters, **decoders):
    for d in list(decoders):
        if decode_cf is False and d in open_backend_dataset_parameters:
            decoders[d] = False
        if decoders[d] is None:
            decoders.pop(d)
    return decoders


def open_dataset(
    filename_or_obj,
    *,
    engine=None,
    chunks=None,
    cache=None,
    decode_cf=None,
    mask_and_scale=None,
    decode_times=None,
    decode_timedelta=None,
    use_cftime=None,
    concat_characters=None,
    decode_coords=None,
    drop_variables=None,
    backend_kwargs=None,
    **kwargs,
):
    """Open and decode a dataset from a file or file-like object.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a netCDF file
        or an OpenDAP URL and opened with python-netCDF4, unless the filename
        ends with .gz, in which case the file is unzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). Byte-strings or file-like
        objects are opened by scipy.io.netcdf (netCDF3) or h5py (netCDF4/HDF).
    engine : str, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        "netcdf4". Options are: {"netcdf4", "scipy", "pydap", "h5netcdf",\
        "pynio", "cfgrib", "pseudonetcdf", "zarr"}.
    chunks : int or dict, optional
        If chunks is provided, it is used to load the new dataset into dask
        arrays. ``chunks=-1`` loads the dataset with dask using a single
        chunk for all arrays. `chunks={}`` loads the dataset with dask using
        engine preferred chunks if exposed by the backend, otherwise with
        a single chunk for all arrays.
        ``chunks='auto'`` will use dask ``auto`` chunking taking into account the
        engine preferred chunks. See dask chunking for more details.
    cache : bool, optional
        If True, cache data is loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False. Does not
        change the behavior of coordinates corresponding to dimensions, which
        always load their data from disk into a ``pandas.Index``.
    decode_cf : bool, optional
        Setting ``decode_cf=False`` will disable ``mask_and_scale``,
        ``decode_times``, ``decode_timedelta``, ``concat_characters``,
        ``decode_coords``.
    mask_and_scale : bool, optional
        If True, array values equal to `_FillValue` are replaced with NA and other
        values are scaled according to the formula `original_values * scale_factor +
        add_offset`, where `_FillValue`, `scale_factor` and `add_offset` are
        taken from variable attributes (if they exist).  If the `_FillValue` or
        `missing_value` attribute contains multiple values, a warning will be
        issued and all array values matching one of the multiple values will
        be replaced by NA. mask_and_scale defaults to True except for the
        pseudonetcdf backend. This keyword may not be supported by all the backends.
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
        This keyword may not be supported by all the backends.
    decode_timedelta : bool, optional
        If True, decode variables and coordinates with time units in
        {"days", "hours", "minutes", "seconds", "milliseconds", "microseconds"}
        into timedelta objects. If False, they remain encoded as numbers.
        If None (default), assume the same value of decode_time.
        This keyword may not be supported by all the backends.
    use_cftime: bool, optional
        Only relevant if encoded dates come from a standard calendar
        (e.g. "gregorian", "proleptic_gregorian", "standard", or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error. This keyword may not be supported by all the backends.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
        This keyword may not be supported by all the backends.
    decode_coords : bool, optional
        If True, decode the 'coordinates' attribute to identify coordinates in
        the resulting dataset. This keyword may not be supported by all the
        backends.
    drop_variables: str or iterable, optional
        A variable or list of variables to exclude from the dataset parsing.
        This may be useful to drop variables with problems or
        inconsistent values.
    backend_kwargs:
        Additional keyword arguments passed on to the engine open function.
    **kwargs: dict
        Additional keyword arguments passed on to the engine open function.
        For example:

        - 'group': path to the netCDF4 group in the given file to open given as
        a str,supported by "netcdf4", "h5netcdf", "zarr".

        - 'lock': resource lock to use when reading data from disk. Only
        relevant when using dask or another form of parallelism. By default,
        appropriate locks are chosen to safely read and write files with the
        currently active dask scheduler. Supported by "netcdf4", "h5netcdf",
        "pynio", "pseudonetcdf", "cfgrib".

        See engine open function for kwargs accepted by each specific engine.


    Returns
    -------
    dataset : Dataset
        The newly created dataset.

    Notes
    -----
    ``open_dataset`` opens the file with read-only access. When you modify
    values of a Dataset, even one linked to files on disk, only the in-memory
    copy you are manipulating in xarray is modified: the original file on disk
    is never touched.

    See Also
    --------
    open_mfdataset
    """

    if cache is None:
        cache = chunks is None

    if backend_kwargs is not None:
        kwargs.update(backend_kwargs)

    if engine is None:
        engine = plugins.guess_engine(filename_or_obj)

    backend = plugins.get_backend(engine)

    decoders = _resolve_decoders_kwargs(
        decode_cf,
        open_backend_dataset_parameters=backend.open_dataset_parameters,
        mask_and_scale=mask_and_scale,
        decode_times=decode_times,
        decode_timedelta=decode_timedelta,
        concat_characters=concat_characters,
        use_cftime=use_cftime,
        decode_coords=decode_coords,
    )

    overwrite_encoded_chunks = kwargs.pop("overwrite_encoded_chunks", None)
    backend_ds = backend.open_dataset(
        filename_or_obj,
        drop_variables=drop_variables,
        **decoders,
        **kwargs,
    )
    ds = _dataset_from_backend_dataset(
        backend_ds,
        filename_or_obj,
        engine,
        chunks,
        cache,
        overwrite_encoded_chunks,
        drop_variables=drop_variables,
        **decoders,
        **kwargs,
    )

    return ds


def _resolve_engine(engine, path_or_file):
    from .api import _get_default_engine
    if path_or_file is None:
        if engine is None:
            engine = "scipy"
    elif isinstance(path_or_file, str):
        if engine is None:
            engine = _get_default_engine(path_or_file)
    else:  # file-like object
        engine = "scipy"
    return engine


def _check_input_consistency(
    engine,
    backend_writer,
    path_or_file,
    compute,
    scheduler,
):
    if path_or_file is None:
        if engine != "scipy":
            raise ValueError(
                "invalid engine for creating bytes with "
                "to_netcdf: %r. Only the default engine "
                "or engine='scipy' is supported" % engine
            )
        if not compute:
            raise NotImplementedError(
                "to_netcdf() with compute=False is not yet implemented when "
                "returning bytes"
            )
    if scheduler and scheduler not in backend_writer.schedulers:
        raise NotImplementedError(
            "Writing netCDF files with the %s backend "
            "is not currently supported with dask's %s "
            "scheduler" % (engine, scheduler)
        )


def prepare_writer(sources, targets, writer=None):
    if not writer:
        writer = ArrayWriter()
    for source, target in zip(sources, targets):
        writer.add(source, target)
    return writer


def to_store(
    dataset: Dataset,
    path_or_file=None,
    *,
    engine: str = None,
    mode: str = "w",
    format: str = None,
    group: str = None,
    compute: bool = True,
    multifile: bool = False,
    encoding: Mapping = None,
    **kwargs,
):
    """This function creates an appropriate datastore for writing a dataset to
    disk as a netCDF file

    See `Dataset.to_netcdf` for full API docs.

    The ``multifile`` argument is only for the private use of save_mfdataset.
    """
    from .api import _finalize_store, _normalize_path, _get_scheduler, _validate_dataset_names, _validate_attrs

    if isinstance(path_or_file, Path):
        path_or_file = str(path_or_file)
        path_or_file = _normalize_path(path_or_file)
    engine = _resolve_engine(engine, path_or_file)

    target = path_or_file if path_or_file is not None else BytesIO()

    if encoding is None:
        encoding = {}
    if format is not None:
        format = format.upper()

    try:
        entrypoint = plugins.get_backend(engine)
        backend_writer = entrypoint.writer
    except KeyError:
        raise ValueError("unrecognized engine for to_netcdf: %r" % engine)

    # handle scheduler specific logic
    have_chunks = any(v.chunks for v in dataset.variables.values())

    if have_chunks:
        scheduler = _get_scheduler()
    else:
        scheduler = None
    if scheduler in ["distributed", "multiprocessing"]:
        autoclose = True
        kwargs["autoclose"] = autoclose
    else:
        autoclose = False

    _check_input_consistency(engine, backend_writer, path_or_file, compute, scheduler)

    # validate Dataset keys, DataArray names, and attr keys/values
    _validate_dataset_names(dataset)
    _validate_attrs(dataset)

    store = backend_writer.open_store(target, mode, format, group, **kwargs)

    # TODO: figure out how to refactor this logic (here and in save_mfdataset)
    # to avoid this mess of conditionals
    try:
        # TODO: allow this work (setting up the file for writing array data)
        # to be parallelized with dask
        sources, targets = store.prepare_store(dataset, encoding=encoding)
        writer = prepare_writer(sources, targets)
        if autoclose:
            store.close()

        if multifile:
            return writer, store

        writes = writer.sync(compute=compute)

        if path_or_file is None:
            store.sync()
            return target.getvalue()
    finally:
        if not multifile and compute:
            store.close()

    if not compute:
        import dask

        return dask.delayed(_finalize_store)(writes, store)
    return None
