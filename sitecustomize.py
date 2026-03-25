import os

# Prevent pytest from auto-loading globally installed plugins that may have
# heavy/optional dependencies not present in this repo's environment.
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
