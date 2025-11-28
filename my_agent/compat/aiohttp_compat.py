"""Compatibility shim to ensure `aiohttp.ClientConnectorDNSError` exists.

Some versions of `google-genai` reference `aiohttp.ClientConnectorDNSError`, but
not all `aiohttp` releases expose that name. This shim defines the attribute as
an alias to `ClientConnectorError` when missing. Import this module before any
package that expects the symbol.
"""
try:
    import aiohttp
    if not hasattr(aiohttp, "ClientConnectorDNSError"):
        # alias to the closest matching exception
        aiohttp.ClientConnectorDNSError = getattr(aiohttp, "ClientConnectorError")
except Exception:
    # If aiohttp is not available yet, ignore — the runtime import will raise later.
    pass
