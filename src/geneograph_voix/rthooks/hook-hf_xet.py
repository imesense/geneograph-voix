# hook-hf_xet.py
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules("hf_xet")
datas = collect_data_files("hf_xet")
