#!/usr/bin/env python3

from functools import cache
import subprocess
import shlex
from pathlib import Path
import logging
import argparse
import os
import glob
import itertools

logger = logging.getLogger(__file__)

LOADER = "#!"
PY_COMMENTS = ("# ", "#", "#")
C_COMMENTS = (" * ", "/*", " */")
COMMENTS = {
    "py": PY_COMMENTS,
    "c": C_COMMENTS,
    "h": C_COMMENTS,
    "cpp": C_COMMENTS,
    "hpp": C_COMMENTS,
}


def cmd_output(cmd: str, cwd: str) -> str:
    cmd_lst = shlex.split(cmd)
    p = subprocess.run(cmd_lst, capture_output=True, check=True, text=True, cwd=cwd)
    if p.returncode != 0:
        raise RuntimeError(
            f"executing command failed: {cmd}:\n"
            " stdout: {p.stdout}\n"
            " stderr: {p.stderr}"
        )
    return p.stdout


def get_git_paths(top_dir: str, dirs: list[str]) -> list[str]:
    out = []
    for path in dirs:
        cmd = f"git ls-files {path}"
        out += cmd_output(cmd, cwd=top_dir).splitlines()
    paths = [p for p in list(dict.fromkeys(out)) if Path(p).is_file()]
    return paths


def get_all_paths(top_dir: str, dirs: list[str]) -> list[str]:
    out = []
    for path in dirs:
        dir = Path(top_dir) / path
        out += [str(p) for p in dir.rglob("*") if p.is_file()]
    paths = list(dict.fromkeys(out))
    return paths


def filter_paths(
    top_dir: str,
    files: list[str],
    includes: list[str],
    excludes: list[str] = [],
) -> list[str]:
    include_set = set(
        itertools.chain(
            *(glob.glob(pat, recursive=True, root_dir=top_dir) for pat in includes)
        )
    )
    exclude_set = set(
        itertools.chain(
            *(glob.glob(pat, recursive=True, root_dir=top_dir) for pat in excludes)
        )
    )
    filter_set = include_set - exclude_set
    top_path = Path(top_dir).resolve()
    files = [str(Path(p).resolve().relative_to(top_path)) for p in files]
    filtered = [p for p in files if p in filter_set]
    return filtered


@cache
def get_license_header(license_file: str, suffix: str) -> list[str]:
    comment, start, end = COMMENTS[suffix]
    header = [
        f"{comment}{l}".rstrip() for l in Path(license_file).read_text().splitlines()
    ]
    return [start, *header, end]


def has_license(lines: list[str], suffix: str) -> bool:
    comment = COMMENTS[suffix][0]
    return any(
        ("copyright" in l or "spdx-license-identifier" in l)
        for l in (l.lower() for l in lines if l.startswith(comment))
    )


def apply_license(license_file: str, fname: str, top: str = ".") -> int:
    path = Path(top) / fname
    suffix = path.suffix[1:]
    header = get_license_header(license_file, suffix)
    in_lines = path.read_text().splitlines()
    if has_license(in_lines, suffix):
        logger.debug(f"Assuming file already licenced: {path}")
        return 0
    out_lines = []
    if LOADER and in_lines and in_lines[0].startswith(LOADER):
        out_lines += [in_lines[0]]
        in_lines = in_lines[1:]
    out_lines += header
    out_lines += in_lines
    tmp_path = Path(f"{path}.tmp")
    with open(tmp_path, "w") as outf:
        for l in out_lines:
            print(l, file=outf)
    tmp_path.replace(path)
    return 1


def check_license(license_file: str, fname: str, top: str = ".") -> bool:
    path = Path(top) / fname
    suffix = path.suffix[1:]
    in_lines = path.read_text().splitlines()
    if has_license(in_lines, suffix):
        return True
    return False


def main():
    TOP_DIR = Path(os.path.relpath(Path(__file__).resolve().parents[2], Path.cwd()))
    DIRS = [Path("src")]
    LICENSE = TOP_DIR / "LICENSE_HEADER"

    INCLUDES = ["**/*.py", "**/*.[ch]", "**/*.[ch]pp"]
    EXCLUDES = []

    parser = argparse.ArgumentParser(
        description="Check/apply LICENSE file to sources",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--apply",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply license, possibly modifying",
    )
    parser.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="check license",
    )
    parser.add_argument(
        "--list",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="only list matched files",
    )
    parser.add_argument(
        "--license",
        type=str,
        default=str(LICENSE),
        help="license file to use",
    )
    parser.add_argument(
        "--top",
        type=str,
        default=str(TOP_DIR),
        help="top level directory",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        type=str,
        default=[str(d) for d in DIRS],
        help="dirs to apply",
    )
    parser.add_argument(
        "--git-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="search git files only, otherwise all files",
    )
    parser.add_argument(
        "--includes",
        nargs="+",
        type=str,
        default=INCLUDES,
        help="includes globs patterns",
    )
    parser.add_argument(
        "--excludes",
        nargs="+",
        type=str,
        default=EXCLUDES,
        help="excludes globs patterns",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="verbose debug mode",
    )
    args = parser.parse_args()

    logging.basicConfig()
    logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.git_files:
        paths = get_git_paths(args.top, args.dirs)
    else:
        paths = get_all_paths(args.top, args.dirs)
    paths = filter_paths(args.top, paths, args.includes, args.excludes)
    if args.list:
        for path in paths:
            print(path)
        raise SystemExit()
    if len(paths) == 0:
        logger.warning("No file found")
        raise SystemExit()
    if args.apply:
        applied = [apply_license(args.license, path, top=args.top) for path in paths]
        count = sum(applied)
        logger.info("Applied %d files: %d changed", len(paths), count)
    elif args.check:
        checks = [check_license(args.license, path, top=args.top) for path in paths]
        failed_count = sum(not c for c in checks)
        if failed_count:
            logger.error(
                "Checked %d files: %d errors, run with --apply to insert license",
                len(paths),
                failed_count,
            )
            raise SystemExit(1)
        else:
            logger.info("Checked %d files", len(paths))


if __name__ == "__main__":
    main()
