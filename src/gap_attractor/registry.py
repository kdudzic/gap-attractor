"""Named, pluggable components: providers, Wikipedia backends and start-page sources.

Each registry maps a config name to a factory. A config value that is not a registered name but a dotted
import path (``my_package.module.Factory``) is imported and used as the factory, so new components can be
added without editing the harness:

    provider.name=my_package.providers.make_vllm            # factory(provider_cfg) -> provider
    wikipedia.backend=my_package.graphs.SyntheticGraph       # factory(wikipedia_cfg, project_root) -> backend
    variant.start_pages_source=my_package.starts.stratified  # factory(cfg) -> list of start titles

Register in code instead with the decorator, e.g. ``@PROVIDERS.register("vllm")``.
"""
from __future__ import annotations

import importlib


class Registry:
    def __init__(self, kind: str):
        self.kind, self._factories = kind, {}

    def register(self, name: str):
        def decorator(factory):
            self._factories[name] = factory
            return factory
        return decorator

    def names(self):
        return tuple(self._factories)

    def is_known(self, name) -> bool:
        return str(name) in self._factories or "." in str(name)

    def get(self, name):
        name = str(name)
        if name in self._factories:
            return self._factories[name]
        if "." in name:
            module_name, _, attribute = name.rpartition(".")
            try:
                return getattr(importlib.import_module(module_name), attribute)
            except (ImportError, AttributeError) as error:
                raise ValueError(f"cannot import {self.kind} {name!r}: {error}") from error
        raise ValueError(f"unknown {self.kind}: {name!r}; expected one of {self.names()} or a dotted import path")


PROVIDERS = Registry("provider")
BACKENDS = Registry("wikipedia backend")
START_PAGE_SOURCES = Registry("start-page source")
