# SPDX-License-Identifier: MPL-2.0
"""Explicit request-local storage binding for isolated service integration."""

from contextlib import contextmanager
from contextvars import ContextVar
import asyncio


_bound_store = ContextVar('instinct_isolated_service_store', default=None)
_bound_config = ContextVar('instinct_isolated_decay_config', default=None)


def bound_store():
    return _bound_store.get()


def bound_config():
    return _bound_config.get()


@contextmanager
def bind_store(store):
    from .dormancy import _ConfigStore
    token = _bound_store.set(store)
    config_token = _bound_config.set(_ConfigStore(store.persist_dir))
    try:
        yield store
    finally:
        _bound_config.reset(config_token)
        _bound_store.reset(token)


async def run_store_call(store, operation, *arguments):
    if bound_store() is store:
        return operation(*arguments)
    return await asyncio.get_running_loop().run_in_executor(None, operation, *arguments)
