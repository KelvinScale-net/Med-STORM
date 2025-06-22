import importlib, pathlib, sys, types

project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# Set package __path__ to project_root/src so submodules can be imported
__path__ = [str(project_root / 'src')]
# Try eager-importing med_storm subpackage to ensure availability
try:
    importlib.import_module('src.med_storm')
except ModuleNotFoundError:
    pass 