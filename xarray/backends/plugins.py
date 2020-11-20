import inspect
import itertools
import typing as T
import warnings
from functools import lru_cache

import pkg_resources  # type: ignore


class BackendEntrypoint:
    __slots__ = ("open_dataset", "open_dataset_parameters")

    def __init__(self, open_dataset, open_dataset_parameters=None):
        self.open_dataset = open_dataset
        self.open_dataset_parameters = open_dataset_parameters


def create_entrypoints_dict(backend_entrypoints):

    # sort and group entrypoints by name
    key_name = lambda ep: ep.name
    backend_entrypoints = sorted(backend_entrypoints, key=key_name)
    backend_entrypoints_grouped =itertools.groupby(
        backend_entrypoints, key=key_name
    )
    # check if there are multiple entrypoints for the same name
    entrypoint_dict = {}
    for name, matches in backend_entrypoints_grouped:
        matches = list(matches)
        matches_len = len(matches)
        entrypoint_dict[name] = matches[0]
        if matches_len > 1:
            selected_module_name = entrypoint_dict[name].module_name
            all_module_names = [e.module_name for e in matches]
            warnings.warn(
                f"\nFound {matches_len} entrypoints for the engine name {name}:"
                f"\n {all_module_names}.\n It will be used: {selected_module_name}.",
            )
    return entrypoint_dict


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
    for name, backend in backend_entrypoints.items():
        backend = backend.load()
        engines[name] = backend
    return engines


def set_parameters(engines):
    for name, backend in engines.items():
        if backend.open_dataset_parameters is None:
            open_dataset = backend.open_dataset
            backend.open_dataset_parameters = detect_parameters(open_dataset)


@lru_cache(maxsize=1)
def detect_engines():
    backend_entrypoints = pkg_resources.iter_entry_points("xarray.backends")
    backend_entrypoints = create_entrypoints_dict(backend_entrypoints)
    engines = load_entrypoints(backend_entrypoints)
    set_parameters(engines)
    return engines
