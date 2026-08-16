class TSReproError(Exception):
    """Base exception for expected TS-Repro failures."""


class ConfigError(TSReproError):
    """A YAML configuration is incomplete or inconsistent."""


class DatasetError(TSReproError):
    """A dataset violates the declared data contract."""


class AdapterError(TSReproError):
    """An adapter cannot produce a valid forecast under the protocol."""


class ManifestError(TSReproError):
    """A sealed experiment directory no longer matches its manifest."""
