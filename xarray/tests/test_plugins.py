import pytest
import pkg_resources

from xarray.backends import plugins

dummy_open_dataset_args = lambda filename_or_obj, *args: None
dummy_open_dataset_kwargs = lambda filename_or_obj, **kwargs: None
dummy_open_dataset = lambda filename_or_obj, *, decoder: None

backend_ep_1 = plugins.BackendEntrypoint(dummy_open_dataset)
backend_ep_2 = plugins.BackendEntrypoint(dummy_open_dataset)

def instantiate_entrypoints(specs):
    # Create the fake entry point definition
    distribution = pkg_resources.Distribution()
    eps = []
    for spec in specs:
        eps.append(
            pkg_resources.EntryPoint.parse(spec, dist=distribution)
        )
    return eps


@pytest.fixture
def dummy_duplicated_entrypoints():
    spec = [
        'engine1 = xarray.tests.test_plugins:backend_ep_1',
        'engine1 = xarray.tests.test_plugins:backend_ep_2',
        'engine2 = xarray.tests.test_plugins:backend_ep_1',
        'engine2 = xarray.tests.test_plugins:backend_ep_2',
    ]
    dummy_duplicated_entrypoints = instantiate_entrypoints(spec)
    return dummy_duplicated_entrypoints


def test_remove_duplicates(dummy_duplicated_entrypoints):
    entrypoints = plugins.remove_duplicates(dummy_duplicated_entrypoints)
    assert len(entrypoints) == 2


def test_remove_duplicates_wargnings(dummy_duplicated_entrypoints):

    with pytest.warns(RuntimeWarning) as record:
        _ = plugins.remove_duplicates(dummy_duplicated_entrypoints)

    assert len(record) == 2
    message0 = str(record[0].message)
    message1 = str(record[1].message)
    assert "entrypoints" in message0
    assert "entrypoints" in message1


def test_create_engines_dict():
    spec = [
        'engine1 = xarray.tests.test_plugins:backend_ep_1',
        'engine2 = xarray.tests.test_plugins:backend_ep_2'
    ]
    entrypoints = instantiate_entrypoints(spec)
    engines = plugins.create_engines_dict(entrypoints)
    assert len(engines) == 2
    assert engines.keys() == set(('engine1', 'engine2'))


def test_set_missing_parameters():
    backend_1 = plugins.BackendEntrypoint(dummy_open_dataset)
    backend_2 = plugins.BackendEntrypoint(dummy_open_dataset, ("filename_or_obj",))
    engines = {"engine_1": backend_1, "engine_2": backend_2}
    plugins.set_missing_parameters(engines)

    assert len(engines) == 2
    engine_1 = engines["engine_1"]
    assert engine_1.open_dataset_parameters == ("filename_or_obj", "decoder")
    engine_2 = engines["engine_2"]
    assert engine_2.open_dataset_parameters == ("filename_or_obj",)


def test_set_missing_parameters_raise_error():

    backend = plugins.BackendEntrypoint(dummy_open_dataset_args)
    with pytest.raises(TypeError):
        plugins.set_missing_parameters({"engine": backend})

    backend = plugins.BackendEntrypoint(dummy_open_dataset_args, ("filename_or_obj", "decoder"))
    plugins.set_missing_parameters({"engine": backend})

    backend = plugins.BackendEntrypoint(dummy_open_dataset_kwargs)
    with pytest.raises(TypeError):
        plugins.set_missing_parameters({"engine": backend})

    backend = plugins.BackendEntrypoint(dummy_open_dataset_kwargs, ("filename_or_obj", "decoder"))
    plugins.set_missing_parameters({"engine": backend})