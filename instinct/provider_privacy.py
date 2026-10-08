# SPDX-License-Identifier: MPL-2.0
"""Explicit provider destinations and non-content embedding validation."""

import ipaddress
import math
from numbers import Real
import os
from urllib.parse import urlsplit


def validated_provider_url(value):
    if not isinstance(value, str) or len(value) > 2048 or any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError('Provider URL must be bounded text.')
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError:
        raise ValueError('Provider URL is malformed; details withheld.') from None
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Provider URL requires HTTP(S), without credentials, query or fragment.')
    try:
        local = parsed.hostname.lower() == 'localhost' or ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        local = parsed.hostname.lower() == 'localhost'
    if not local and os.environ.get('INSTINCT_ALLOW_REMOTE_PROVIDERS') != '1':
        raise ValueError('Non-loopback providers require INSTINCT_ALLOW_REMOTE_PROVIDERS=1 after privacy review.')
    if not local and parsed.scheme != 'https' and os.environ.get('INSTINCT_ALLOW_INSECURE_REMOTE_HTTP') != '1':
        raise ValueError('Remote providers require HTTPS or explicit INSTINCT_ALLOW_INSECURE_REMOTE_HTTP=1.')
    return value.rstrip('/')


def valid_query_embedding(vectors):
    try:
        if isinstance(vectors, (str, bytes, dict)) or len(vectors) != 1:
            return False
        vector = vectors[0]
        if isinstance(vector, (str, bytes, dict)) or not 1 <= len(vector) <= 65536:
            return False
        return all(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)) for value in vector)
    except (TypeError, ValueError, OverflowError, IndexError):
        return False
