import numpy as np

from .. import conventions
from ..core import indexing
from ..core import dataset
from ..core.utils import Frozen, FrozenDict, close_on_error
from ..core.variable import Variable

from .common import AbstractDataStore, BackendArray
from .locks import SerializableLock, ensure_lock

# FIXME: Add a dedicated lock, even if ecCodes is supposed to be thread-safe
#   in most circumstances. See:
#       https://confluence.ecmwf.int/display/ECC/Frequently+Asked+Questions
ECCODES_LOCK = SerializableLock()


class CfGribArrayWrapper(BackendArray):
    def __init__(self, datastore, array):
        self.datastore = datastore
        self.shape = array.shape
        self.dtype = array.dtype
        self.array = array

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER, self._getitem
        )

    def _getitem(self, key):
        with self.datastore.lock:
            return self.array[key]


class CfGribDataStore(AbstractDataStore):
    """
    Implements the ``xr.AbstractDataStore`` read-only API for a GRIB file.
    """

    def __init__(self, filename, lock=None, **backend_kwargs):
        import cfgrib

        if lock is None:
            lock = ECCODES_LOCK
        self.lock = ensure_lock(lock)
        self.ds = cfgrib.open_file(filename, **backend_kwargs)

    def open_store_variable(self, name, var):
        if isinstance(var.data, np.ndarray):
            data = var.data
        else:
            wrapped_array = CfGribArrayWrapper(self, var.data)
            data = indexing.LazilyOuterIndexedArray(wrapped_array)

        encoding = self.ds.encoding.copy()
        encoding["original_shape"] = var.data.shape

        return Variable(var.dimensions, data, var.attributes, encoding)

    def get_variables(self):
        return FrozenDict(
            (k, self.open_store_variable(k, v)) for k, v in self.ds.variables.items()
        )

    def get_attrs(self):
        return Frozen(self.ds.attributes)

    def get_dimensions(self):
        return Frozen(self.ds.dimensions)

    def get_encoding(self):
        dims = self.get_dimensions()
        encoding = {"unlimited_dims": {k for k, v in dims.items() if v is None}}
        return encoding


def open_cfgrib_(
        filename_or_obj,
        xr_decoders,
        **kwargs
    ):
    store = CfGribDataStore(filename_or_obj, **kwargs)

    with close_on_error(store):
        vars, attrs = store.load()
        extra_coords = set()
        file_obj = store
        encoding = store.get_encoding()

        vars, attrs, coord_names = conventions.decode_cf_variables(
            vars,
            attrs,
            **xr_decoders,
        )

        ds = dataset.Dataset(vars, attrs=attrs)
        ds = ds.set_coords(coord_names.union(extra_coords).intersection(vars))
        ds._file_obj = file_obj
        ds.encoding = encoding

    return ds



