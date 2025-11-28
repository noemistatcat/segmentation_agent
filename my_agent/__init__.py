# Ensure compatibility shims are applied before importing agents/tools
try:
    from .compat import aiohttp_compat  # applies runtime alias if needed
except Exception:
    # If compat package is missing or fails, allow import to continue; errors will surface later.
    pass
from . import agent
