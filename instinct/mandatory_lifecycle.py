# SPDX-License-Identifier: MPL-2.0
"""Connection and explicit client-event restoration of existing mandatories."""

from threading import Lock

from .nudge import renew_mandatory


class MandatoryLifecycle:
    """Keep a renewal pending until a writable store successfully applies it."""

    def __init__(self):
        self._pending = True
        self._lock = Lock()

    def request(self):
        with self._lock:
            self._pending = True

    def restore_if_pending(self, store) -> bool:
        with self._lock:
            if not self._pending:
                return True
            if getattr(store, '_read_only', False):
                return False
            renew_mandatory(store)
            self._pending = False
            return True
