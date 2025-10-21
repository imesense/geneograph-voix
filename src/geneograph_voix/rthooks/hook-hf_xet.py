# hook-hf_xet.py
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules("hf_xet") + collect_submodules("pyxet")
datas = collect_data_files("hf_xet") + collect_data_files("pyxet")
