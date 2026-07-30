# __init__.py — marks adapters/ as a Python package
# Imports are done lazily inside each adapter's generate() call,
# so importing this package does NOT pull in httpx, torch, etc.
