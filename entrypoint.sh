#!/bin/sh
# Update ldconfig cache if AXCL libraries are mounted (needed by axengine's
# ctypes.util.find_library which uses ldconfig -p, not LD_LIBRARY_PATH).
if [ -d /usr/lib/axcl ]; then
    ldconfig 2>/dev/null || true
fi

exec setpriv --reuid=appuser --regid=appuser --init-groups python -m vision2mqtt "$@"
