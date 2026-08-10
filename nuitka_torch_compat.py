"""Run Nuitka with a narrow compatibility fix for Torch 2.10 on Windows.

Nuitka 4.1.3 evaluates every Torch module named ``config`` or ``_config``.
Torch 2.10 includes optional config modules that can raise while Nuitka imports
them in its isolated build-time query. This wrapper changes that query to skip
only modules that cannot be imported, while preserving all successfully loaded
Torch configuration modules.

The installed Nuitka package is never modified. A patched private copy is kept
under ``build/.nuitka-compat-4.1.3`` and used only by the child build process.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys


SUPPORTED_NUITKA_VERSION = "4.1.3"
PATCH_REVISION = b"orbit-torch-config-import-guard-v2"

ORIGINAL_SETUP = "      - 'import importlib'\n"
PATCHED_SETUP = """      - 'import importlib'
      - |
        def _nuitka_import_torch_config_module(module_name):
          try:
            return importlib.import_module(module_name)
          except Exception:
            return None
"""

ORIGINAL_DECLARATION = (
    "      'torch_config_modules': 'dict((m,importlib.import_module(m)."
    "_compile_ignored_keys) for m in torch_config_module_candidates if "
    "hasattr(importlib.import_module(m), \"_compile_ignored_keys\"))'"
)
PATCHED_DECLARATION = (
    "      'torch_config_modules': 'dict((m,module._compile_ignored_keys) "
    "for m in torch_config_module_candidates for module in "
    "(_nuitka_import_torch_config_module(m),) if module is not None and "
    "hasattr(module, \"_compile_ignored_keys\"))'"
)


def _find_nuitka_package() -> Path:
    version = importlib.metadata.version("Nuitka")
    if version != SUPPORTED_NUITKA_VERSION:
        raise RuntimeError(
            f"This compatibility wrapper supports Nuitka "
            f"{SUPPORTED_NUITKA_VERSION}, but {version} is installed."
        )

    spec = importlib.util.find_spec("nuitka")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Could not locate the installed Nuitka package.")

    return Path(next(iter(spec.submodule_search_locations))).resolve()


def _patch_package(source_package: Path, destination_root: Path) -> None:
    source_config = (
        source_package
        / "plugins"
        / "standard"
        / "standard.nuitka-package.config.yml"
    )
    if not source_config.is_file():
        raise RuntimeError(f"Nuitka package configuration was not found: {source_config}")

    source_bytes = source_config.read_bytes()
    signature = hashlib.sha256(source_bytes + PATCH_REVISION).hexdigest()
    marker = destination_root / ".orbit-patch-signature"
    destination_package = destination_root / "nuitka"

    if (
        destination_package.is_dir()
        and marker.is_file()
        and marker.read_text(encoding="ascii").strip() == signature
    ):
        return

    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_package, destination_package)

    destination_config = (
        destination_package
        / "plugins"
        / "standard"
        / "standard.nuitka-package.config.yml"
    )
    config_text = destination_config.read_text(encoding="utf-8")

    module_start = config_text.find("- module-name: 'torch.utils._config_module'")
    if module_start < 0:
        raise RuntimeError("The Nuitka Torch configuration entry was not found.")
    module_end = config_text.find("\n- module-name:", module_start + 1)
    if module_end < 0:
        raise RuntimeError("The end of the Nuitka Torch configuration entry was not found.")

    module_config = config_text[module_start:module_end]
    if module_config.count(ORIGINAL_SETUP) != 1:
        raise RuntimeError("The expected Nuitka Torch setup block was not found exactly once.")
    if module_config.count(ORIGINAL_DECLARATION) != 1:
        raise RuntimeError(
            "The expected Nuitka Torch variable declaration was not found exactly once."
        )

    module_config = module_config.replace(ORIGINAL_SETUP, PATCHED_SETUP, 1)
    module_config = module_config.replace(
        ORIGINAL_DECLARATION, PATCHED_DECLARATION, 1
    )
    config_text = (
        config_text[:module_start] + module_config + config_text[module_end:]
    )

    # Nuitka validates any entry whose embedded checksum changed. Recalculate
    # this one so the local compatibility edit remains a normal trusted entry
    # and does not trigger an unnecessary jsonschema download.
    from nuitka.Tracing import general
    from nuitka.utils.Yaml import getYamlDataHash, parseYaml

    parsed_config = parseYaml(
        logger=general,
        data=config_text.encode("utf-8"),
        error_message="Could not parse the patched Nuitka package configuration.",
    )
    parsed_module = next(
        item.copy()
        for item in parsed_config
        if item.get("module-name") == "torch.utils._config_module"
    )
    del parsed_module["module-name"]
    checksum = getYamlDataHash(parsed_module)

    header_end = config_text.find("\n", module_start)
    header = config_text[module_start:header_end]
    if "# checksum: " not in header:
        raise RuntimeError("The Nuitka Torch configuration checksum was not found.")
    patched_header = header.split("# checksum: ", 1)[0] + "# checksum: " + checksum
    config_text = (
        config_text[:module_start] + patched_header + config_text[header_end:]
    )

    destination_config.write_text(config_text, encoding="utf-8")
    marker.write_text(signature + "\n", encoding="ascii")


def main() -> int:
    try:
        source_package = _find_nuitka_package()
        repository_root = Path(__file__).resolve().parent
        destination_root = (
            repository_root / "build" / f".nuitka-compat-{SUPPORTED_NUITKA_VERSION}"
        )
        _patch_package(source_package, destination_root)
    except Exception as exc:
        print(f"ERROR: Could not prepare the Nuitka Torch compatibility fix: {exc}")
        return 1

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(destination_root)
    if existing_pythonpath:
        environment["PYTHONPATH"] += os.pathsep + existing_pythonpath

    print(
        f"ORBIT build: using Nuitka {SUPPORTED_NUITKA_VERSION} with the "
        "Torch config evaluation fix.",
        flush=True,
    )
    command = [sys.executable, "-m", "nuitka", *sys.argv[1:]]
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
