# GeneoGraph Voice Indexer

## Requirements

- Python 3.10

## Build

### Windows

Configure virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt --cache-dir .pip
pip install -r src\geneograph_voix\requirements.txt --cache-dir .pip
pip install -r src\geneograph_voix\requirements_cuda.txt --cache-dir .pip
```

Pack modules:

```powershell
pyinstaller src\geneograph_voix\geneograph_voix.spec
```

Deactivate virtual environment:

```powershell
deactivate
```

### macOS

Configure virtual environment:

```sh
python3 -m venv .venv
chmod +x .venv/bin/activate
./.venv/bin/activate
```

Install dependencies:

```sh
python3 -m pip install -r requirements.txt --cache-dir .pip
python3 -m pip install -r src/geneograph_voix/requirements.txt --cache-dir .pip
```

Pack modules:

```sh
pyinstaller src/geneograph_voix/geneograph_voix.spec
```

Deactivate virtual environment:

```sh
deactivate
```

## License

Contents of this repository licensed under terms of the __GNU GPL 3 license__ unless otherwise specified. See [this](./LICENSE) file for details
