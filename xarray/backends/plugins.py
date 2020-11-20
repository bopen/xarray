import inspect
import itertools
import typing as T
import warnings
from functools import lru_cache

import pkg_resources

class BackendEntrypoint:
    __slots__ = ("open_dataset", "open_dataset_parameters")

    def __init__(self, open_dataset, open_dataset_parameters=None):
        self.open_dataset = open_dataset
        self.open_dataset_parameters = open_dataset_parameters


def remove_duplicates(backend_entrypoints):

    # sort and group entrypoints by name
    key_name = lambda ep: ep.name
    backend_entrypoints = sorted(backend_entrypoints, key=key_name)
    backend_entrypoints_grouped =itertools.groupby(
        backend_entrypoints, key=key_name
    )
    # check if there are multiple entrypoints for the same name
    unique_backend_entrypoints = []
    for name, matches in backend_entrypoints_grouped:
        matches = list(matches)
        unique_backend_entrypoints.append([name, matches[0]])
        matches_len = len(matches)
        if matches_len > 1:
            selected_module_name = matches[0].module_name
            all_module_names = [e.module_name for e in matches]
            warnings.warn(
                f"\nFound {matches_len} entrypoints for the engine name {name}:"
                f"\n {all_module_names}.\n It will be used: {selected_module_name}.",
            )
    return unique_backend_entrypoints


def detect_parameters(open_dataset):
    signature = inspect.signature(open_dataset)
    parameters = signature.parameters
    for name, param in parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            raise TypeError(
                f"All the parameters in {open_dataset!r} signature should be explicit. "
                "*args and **kwargs is not supported"
            )
    return set(parameters)


def load_entrypoints(backend_entrypoints):
    engines: T.Dict[str, T.Dict[str, T.Any]] = {}
    for name, backend in backend_entrypoints:
        backend = backend.load()
        engines[name] = backend
    return engines


def set_missing_parameters(engines):
    for name, backend in engines.items():
        if backend.open_dataset_parameters is None:
            open_dataset = backend.open_dataset
            backend.open_dataset_parameters = detect_parameters(open_dataset)


@lru_cache(maxsize=1)
def detect_engines():
    entrypoints = pkg_resources.iter_entry_points("xarray.backends")
    backend_entrypoints = remove_duplicates(entrypoints)
    engines = load_entrypoints(backend_entrypoints)
    set_missing_parameters(engines)
    return engines
