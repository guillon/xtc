from pathlib import Path
from subprocess import CompletedProcess

import pytest

from xtc.utils import ext_tools


@pytest.mark.parametrize("search_variable", ["LD_LIBRARY_PATH", "LIBRARY_PATH"])
def test_get_library_path_uses_environment_search_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, search_variable: str
) -> None:
    library_name = "libomp.so.5"
    library = tmp_path / library_name
    library.touch()

    monkeypatch.setattr(ext_tools.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        ext_tools.ctypes.util, "find_library", lambda _name: library_name
    )
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("LIBRARY_PATH", raising=False)
    monkeypatch.setenv(search_variable, str(tmp_path))
    monkeypatch.setattr(ext_tools.shutil, "which", lambda _name: None)

    assert ext_tools.get_library_path("omp") == str(library)


def test_get_library_path_falls_back_to_ldconfig(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library_name = "libomp.so.5"
    library = "/usr/lib/x86_64-linux-gnu/libomp.so.5"

    monkeypatch.setattr(ext_tools.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        ext_tools.ctypes.util, "find_library", lambda _name: library_name
    )
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("LIBRARY_PATH", raising=False)
    monkeypatch.setattr(ext_tools.shutil, "which", lambda _name: "/sbin/ldconfig")
    monkeypatch.setattr(
        ext_tools.subprocess,
        "run",
        lambda *_args, **_kwargs: CompletedProcess(
            args=["/sbin/ldconfig", "-p"],
            returncode=0,
            stdout=f"\t{library_name} (libc6,x86-64) => {library}\n",
        ),
    )

    assert ext_tools.get_library_path("omp") == library
