#!/usr/bin/env python3
"""GATE-9, installed in a minute: time each published install route to a verdict on the demo.

    python3 scripts/install_smoke.py [--route pip|pipx|uvx|installer|binary ...] [--version V]
        [--runs 3] [--budget 60] [--step-timeout 600] [--release-base URL] [--out FILE]
        [--workdir DIR]
    python3 scripts/install_smoke.py --check DIR [--version V]     # the gate: the CI records

Each run of each route starts in a fresh HOME (so fresh pip, pipx and uv caches and install
directories) and goes from nothing to a verdict, as the public README tells a new user:

  pip        python3 -m venv venv; venv/bin/python -m pip install 'courant-dev[mcp]==V'
  pipx       pipx install courant-dev==V
  uvx        uvx --from courant-dev==V courant ... (it installs on first use)
  installer  curl releases/latest/download/install.sh; sh install.sh, which must print
             "signature verified" (SHA256SUMS against the release key) and checks the
             binary's checksum; then `courant version` must name V
  binary     curl courant-<os>-<arch> and its .sha256 of release vV; check the checksum
             (refused on a mismatch) and chmod +x

then, on every route, `courant demo --keep demo` (exit 0; the healthy cavity accepted, rule
VER-5.6 grade A; the broken one rejected, rule VER-5.3 grade C) and `courant verify` on the kept
icoFoam_cavity__cfl_blowup run, the README's example (exit 2, `VERDICT   reject   (rule
VER-5.3)`). A run's time is the sum of its steps' wall times, from the first download or install
to the verify's verdict: later than the first verdict, which the demo prints. The bootstrap of
pipx or uv is not part of a route. Exit 0 only when every run of every route passed in under
the budget; 1 otherwise; 2 on a usage error. A missing tool or an unpublished platform is a
failure with its reason, never a skip. `--out` writes courant.install-smoke/1 JSON, after each
route, so a job killed later keeps what was measured.

`--check DIR` is the gate (G-INSTALL): it measures nothing here, and passes only when DIR holds,
for each of linux-x86_64 and darwin-arm64, a record written on a GitHub Actions runner for this
version with every route passed, at least three runs each, under a budget of at most 60 s.

Standard library only, python3 >= 3.9, and no import of courant: it measures an install, so it
must run on a machine where Courant is not installed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "courant.install-smoke/1"
REPO = "courantdev/courant"
DIST = "courant-dev"
LATEST_BASE = f"https://github.com/{REPO}/releases/latest/download"
ROUTES = ("pip", "pipx", "uvx", "installer", "binary")
#: The tool each route needs on PATH before it starts.
TOOL = {"pip": "python3", "pipx": "pipx", "uvx": "uvx", "installer": "curl", "binary": "curl"}
#: The binaries a release publishes and packaging/install.sh installs.
PUBLISHED_TARGETS = ("linux-x86_64", "linux-arm64", "darwin-arm64")
#: The platforms GATE-9 names; the gate needs a CI record of each.
REQUIRED_TARGETS = ("linux-x86_64", "darwin-arm64")
MIN_RUNS = 3
DEFAULT_BUDGET_S = 60.0
DEFAULT_RUNS = 3
DEFAULT_STEP_TIMEOUT_S = 600.0

#: The demo run the README verifies by hand, kept by `courant demo --keep demo`.
KEPT = "icoFoam_cavity__cfl_blowup"
VERIFY_ARGS = ["verify", f"demo/{KEPT}/{KEPT}.log.gz", "--case-json",
               f"demo/{KEPT}/{KEPT}.case.json"]
DEMO_ARGS = ["demo", "--keep", "demo"]
DEMO_EXPECT = ("accept  (rule VER-5.6, grade A)", "reject  (rule VER-5.3, grade C)")
VERIFY_EXPECT = ("VERDICT   reject   (rule VER-5.3)",)
SIGNATURE_EXPECT = "signature verified"

#: Variables that would move pip's, pipx's or uv's state, or the installer's target, out of
#: the fresh HOME. Everything else (the proxy, the CA bundle) is the machine's and stays.
RELOCATING = {
    "PIPX_HOME", "PIPX_BIN_DIR", "PIPX_MAN_DIR", "PIPX_SHARED_LIBS", "PIPX_DEFAULT_PYTHON",
    "PIP_CACHE_DIR", "PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PIP_NO_CACHE_DIR", "PIP_FIND_LINKS",
    "UV_CACHE_DIR", "UV_TOOL_DIR", "UV_TOOL_BIN_DIR", "UV_NO_CACHE", "UV_OFFLINE",
    "UV_FIND_LINKS", "UV_PYTHON_INSTALL_DIR",
    "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_BIN_HOME", "XDG_CONFIG_HOME", "XDG_STATE_HOME",
    "VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE",
}
OUTPUT_TAIL_FAILED = 3000
OUTPUT_TAIL_PASSED = 1200


@dataclass
class Step:
    name: str
    argv: list[str]
    expect_code: int = 0
    expect_text: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


class UsageError(Exception):
    pass


# ---------------------------------------------------------------------------- what to run

def normalise_version(v: str) -> str:
    v = v.strip()
    return v[1:] if v[:1] in ("v", "V") else v


def default_version(root: Path = ROOT) -> str:
    """The checkout's own version, read as text: the harness never imports courant."""
    try:
        text = (root / "courant" / "__init__.py").read_text()
    except OSError as exc:
        raise UsageError(f"cannot read courant/__init__.py ({exc}); pass --version") from exc
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not m:
        raise UsageError("no __version__ in courant/__init__.py; pass --version")
    return m.group(1)


def release_url(version: str) -> str:
    return f"https://github.com/{REPO}/releases/download/v{normalise_version(version)}"


def target_of(system: str, machine: str) -> str:
    os_ = {"Linux": "linux", "Darwin": "darwin"}.get(system, system.lower())
    arch = {"x86_64": "x86_64", "amd64": "x86_64", "AMD64": "x86_64",
            "arm64": "arm64", "aarch64": "arm64"}.get(machine, machine.lower())
    return f"{os_}-{arch}"


def target() -> str:
    """This machine's release asset, named as packaging/install.sh names it."""
    t = target_of(platform.system(), platform.machine())
    if t not in PUBLISHED_TARGETS:
        raise ValueError(f"no Courant binary is published for {t} "
                         f"(published: {', '.join(PUBLISHED_TARGETS)})")
    return t


def plan(route: str, version: str, *, home: Path, release_base: str,
         tools: dict[str, str]) -> list[Step]:
    """The steps of one run of `route`, in order. Relative paths are under `home`, the cwd."""
    version = normalise_version(version)
    pin = f"{DIST}=={version}"
    base = release_base.rstrip("/")
    local_courant = str(home / ".local" / "bin" / "courant")

    def demo_verify(courant: list[str]) -> list[Step]:
        return [Step("demo", courant + DEMO_ARGS, 0, DEMO_EXPECT),
                Step("verify", courant + VERIFY_ARGS, 2, VERIFY_EXPECT)]

    if route == "pip":
        venv = home / "venv" / "bin"
        return [Step("venv", [tools["python3"], "-m", "venv", "venv"]),
                Step("install", [str(venv / "python"), "-m", "pip", "install",
                                 f"{DIST}[mcp]=={version}"])
                ] + demo_verify([str(venv / "courant")])
    if route == "pipx":
        return [Step("install", [tools["pipx"], "install", pin])] + demo_verify([local_courant])
    if route == "uvx":
        return demo_verify([tools["uvx"], "--from", pin, "courant"])
    if route == "installer":
        # The README's command: the latest release's installer, no version named.
        return [Step("download", [tools["curl"], "-fsSL", "-o", "install.sh",
                                  f"{LATEST_BASE}/install.sh"]),
                Step("install", ["sh", "install.sh"], 0, (SIGNATURE_EXPECT,)),
                Step("version", [local_courant, "version"], 0, (f"courant {version} ",)),
                ] + demo_verify([local_courant])
    if route == "binary":
        asset = f"courant-{target()}"
        return [Step("download", [tools["curl"], "-fsSL", "--fail-early",
                                  "-o", asset, f"{base}/{asset}",
                                  "-o", f"{asset}.sha256", f"{base}/{asset}.sha256"]),
                Step("checksum", [sys.executable, str(Path(__file__).resolve()), "_sha256",
                                  asset, f"{asset}.sha256", "--make-executable"])
                ] + demo_verify([str(home / asset)])
    raise ValueError(f"unknown route {route!r}")


def step_env(home: Path, base: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in base.items()
           if k not in RELOCATING and not k.startswith("COURANT_")}
    env["HOME"] = str(home)
    env["PATH"] = os.pathsep.join([str(home / ".local" / "bin"), base.get("PATH", "")])
    return env


# ------------------------------------------------------------------------------ running it

def summarise(seconds: list[float]) -> dict:
    if not seconds:
        return {"median_s": None, "slowest_s": None}
    return {"median_s": round(statistics.median(seconds), 3),
            "slowest_s": round(max(seconds), 3)}


def run_step(step: Step, home: Path, env: dict[str, str], timeout: float) -> dict:
    rec: dict = {"name": step.name, "argv": step.argv, "expect_code": step.expect_code}
    if step.env:
        rec["env"] = step.env
    failure = None
    t0 = time.monotonic()
    try:
        proc = subprocess.run(step.argv, cwd=home, env={**env, **step.env},
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=timeout, check=False)
        out = proc.stdout.decode("utf-8", "replace")
        code: int | None = proc.returncode
    except subprocess.TimeoutExpired as exc:
        out = (exc.output or b"").decode("utf-8", "replace")
        code, failure = None, f"did not finish: timed out after {timeout:g} s"
    except OSError as exc:                   # the command itself could not start
        out = f"{type(exc).__name__}: {exc}"
        code, failure = None, "could not start"
    rec["seconds"] = round(time.monotonic() - t0, 3)
    rec["code"] = code
    why = []
    if failure:
        why.append(failure)
    elif code != step.expect_code:
        why.append(f"exit {code}, expected {step.expect_code}")
    why += [f"output lacks {t!r}" for t in step.expect_text if t not in out]
    rec["ok"] = not why
    if why:
        rec["why"] = "; ".join(why)
    rec["output"] = out[-(OUTPUT_TAIL_FAILED if why else OUTPUT_TAIL_PASSED):]
    return rec


def run_route(route: str, *, version: str, release_base: str, runs: int, budget: float,
              workdir: Path, step_timeout: float, base_env: dict[str, str]) -> dict:
    res: dict = {"route": route, "runs": []}
    tool = shutil.which(TOOL[route], path=base_env.get("PATH", ""))
    if tool is None:
        res.update(passed=False, error=f"{TOOL[route]} is not on PATH", completed_runs=0,
                   **summarise([]))
        return res
    if route in ("installer", "binary"):
        try:
            res["target"] = target()
        except ValueError as exc:
            res.update(passed=False, error=str(exc), completed_runs=0, **summarise([]))
            return res
    tools = {**{t: t for t in TOOL.values()}, TOOL[route]: tool}
    for i in range(runs):
        home = Path(tempfile.mkdtemp(prefix=f"{route}-{i + 1}-", dir=workdir))
        env = step_env(home, base_env)
        steps = plan(route, version, home=home, release_base=release_base, tools=tools)
        recs = []
        for step in steps:
            rec = run_step(step, home, env, step_timeout)
            recs.append(rec)
            if not rec["ok"]:
                break
        seconds = round(sum(r["seconds"] for r in recs), 3)
        completed = len(recs) == len(steps) and all(r["ok"] for r in recs)
        run = {"run": i + 1, "home": str(home), "steps": recs, "seconds": seconds,
               "completed": completed, "under_budget": completed and seconds < budget}
        run["passed"] = run["under_budget"]
        res["runs"].append(run)
    done = [r["seconds"] for r in res["runs"] if r["completed"]]
    res["completed_runs"] = len(done)
    res.update(summarise(done))              # over completed runs: a quick failure is no time
    res["passed"] = all(r["passed"] for r in res["runs"])
    return res


def _version_of(argv: list[str], env: dict[str, str]) -> str | None:
    exe = shutil.which(argv[0], path=env.get("PATH", ""))
    if exe is None:
        return None
    try:
        p = subprocess.run([exe, *argv[1:]], capture_output=True, text=True, timeout=30,
                           env=env, stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return ((p.stdout or p.stderr).strip().splitlines() or [""])[0][:120]


def _os_release() -> str | None:
    try:
        return platform.freedesktop_os_release().get("PRETTY_NAME")  # python 3.10+
    except (AttributeError, OSError):
        mac = platform.mac_ver()[0]
        return f"macOS {mac}" if mac else None


def environment(env: dict[str, str]) -> dict:
    return {
        "os": platform.system(),
        "machine": platform.machine(),
        "os_release": _os_release(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
        "tools": {"python3": _version_of(["python3", "--version"], env),
                  "pipx": _version_of(["pipx", "--version"], env),
                  "uv": _version_of(["uv", "--version"], env),
                  "curl": _version_of(["curl", "--version"], env),
                  "openssl": _version_of(["openssl", "version"], env)},
        "runner": {k: env[k] for k in ("GITHUB_ACTIONS", "RUNNER_OS", "RUNNER_ARCH",
                                       "RUNNER_ENVIRONMENT", "ImageOS", "ImageVersion")
                   if k in env},
        # datetime.UTC is 3.11+; the harness runs on any python3 a runner has
        "recorded_at": _dt.datetime.now(_dt.timezone.utc).strftime(  # noqa: UP017
            "%Y-%m-%dT%H:%M:%SZ"),
    }


# ----------------------------------------------------------------------- the checksum step

def check_sha256(path: Path, sumfile: Path, make_executable: bool) -> int:
    try:
        fields = sumfile.read_text().split()
        expected = fields[0].lower() if fields else ""
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            print(f"no SHA-256 in {sumfile.name}: refusing to run {path.name}")
            return 1
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        print(f"cannot check {path.name}: {exc}")
        return 1
    if actual != expected:
        print(f"checksum mismatch: {path.name}\n  expected  {expected}\n  actual    {actual}\n"
              "refusing to run it")
        return 1
    if make_executable:
        path.chmod(path.stat().st_mode | 0o111)
    print(f"{path.name}: sha256 {actual} matches {sumfile.name}")
    return 0


# ---------------------------------------------------------------- the gate: the CI records

def record_problems(doc: dict, version: str) -> list[str]:
    """Why a record does not show GATE-9 for its platform; empty when it does."""
    p = []
    if doc.get("schema") != SCHEMA:
        return [f"not a {SCHEMA} record"]
    if (doc.get("environment") or {}).get("runner", {}).get("GITHUB_ACTIONS") != "true":
        p.append("not on a CI runner (no GITHUB_ACTIONS=true)")
    if doc.get("version") != version:
        p.append(f"version {doc.get('version')!r}, not {version!r}")
    budget = doc.get("budget_s")
    if not isinstance(budget, (int, float)) or budget > DEFAULT_BUDGET_S:
        p.append(f"budget {budget!r} s, more than {DEFAULT_BUDGET_S:g} s")
    routes = doc.get("routes") or {}
    for name in ROUTES:
        r = routes.get(name)
        if r is None:
            p.append(f"route {name} not measured")
            continue
        runs = r.get("runs") or []
        if len(runs) < MIN_RUNS:
            p.append(f"route {name}: {len(runs)} runs, fewer than {MIN_RUNS}")
        if not (r.get("passed") is True and runs and all(x.get("passed") is True for x in runs)):
            p.append(f"route {name} did not pass")
        elif isinstance(budget, (int, float)) and not all(
                isinstance(x.get("seconds"), (int, float)) and x["seconds"] < budget
                for x in runs):
            p.append(f"route {name}: a run at or over {budget:g} s")
    if doc.get("passed") is not True:
        p.append("the record says it did not pass")
    return p


def check_records(directory: Path, version: str) -> int:
    found: dict[str, list[tuple[str, list[str]]]] = {}
    for f in sorted(directory.rglob("*.json")) if directory.is_dir() else []:
        try:
            doc = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            continue
        env = doc.get("environment") or {}
        t = target_of(str(env.get("os", "")), str(env.get("machine", "")))
        found.setdefault(t, []).append((str(f.relative_to(directory)),
                                         record_problems(doc, version)))
    if not found:
        print(f"install_smoke --check: no {SCHEMA} record in {directory}")
        return 1
    ok = True
    for t in REQUIRED_TARGETS:
        recs = found.get(t, [])
        good = [name for name, probs in recs if not probs]
        if good:
            print(f"{t}: GATE-9 shown by {good[0]}")
            continue
        ok = False
        if not recs:
            print(f"{t}: MISSING, no record of this platform")
        for name, probs in recs:
            print(f"{t}: {name}: " + "; ".join(probs))
    print(f"install_smoke --check: courant-dev {version}: "
          + ("every required platform passed on a CI runner" if ok else "FAILED"))
    return 0 if ok else 1


# ------------------------------------------------------------------------------------ main

def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="install_smoke", description=__doc__.splitlines()[0])
    ap.add_argument("--route", action="append", choices=ROUTES,
                    help="a route to measure (repeatable; default: all)")
    ap.add_argument("--version", help="the published version (default: this checkout's)")
    ap.add_argument("--release-base", help="the binary route's download URL (default: the "
                    f"GitHub release of {REPO} for the version)")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET_S, help="seconds")
    ap.add_argument("--step-timeout", type=float, default=DEFAULT_STEP_TIMEOUT_S)
    ap.add_argument("--out", type=Path, help="write the courant.install-smoke/1 JSON here")
    ap.add_argument("--workdir", type=Path,
                    help="keep every run's HOME under DIR (default: a temporary directory, "
                         "removed at the end)")
    ap.add_argument("--check", type=Path, metavar="DIR",
                    help="measure nothing: check the CI records in DIR (the gate)")
    return ap


def _write(path: Path | None, doc: dict) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["_sha256"]:
        rest = [a for a in argv[1:] if a != "--make-executable"]
        if len(rest) != 2:
            print("usage: install_smoke.py _sha256 FILE SUMFILE [--make-executable]",
                  file=sys.stderr)
            return 2
        return check_sha256(Path(rest[0]), Path(rest[1]), "--make-executable" in argv)
    ap = _parser()
    args = ap.parse_args(argv)
    if args.runs < 1:
        ap.error("--runs must be at least 1")
    if args.budget <= 0 or args.step_timeout <= 0:
        ap.error("--budget and --step-timeout must be positive")
    try:
        version = normalise_version(default_version() if args.version is None
                                    else args.version)
    except UsageError as exc:
        ap.error(str(exc))
    if not version:
        ap.error("--version is empty")
    if args.check is not None:
        return check_records(args.check, version)
    release_base = args.release_base or release_url(version)
    routes = list(dict.fromkeys(args.route or ROUTES))
    base_env = dict(os.environ)

    keep = args.workdir is not None
    workdir = (args.workdir.resolve() if keep
               else Path(tempfile.mkdtemp(prefix="courant-install-smoke-")))
    workdir.mkdir(parents=True, exist_ok=True)
    doc = {"schema": SCHEMA, "version": version, "release_base": release_base,
           "latest_base": LATEST_BASE, "budget_s": float(args.budget),
           "runs_per_route": args.runs, "step_timeout_s": float(args.step_timeout),
           "environment": environment(base_env), "routes": {}, "passed": False,
           "complete": False}
    try:
        for route in routes:
            res = run_route(route, version=version, release_base=release_base, runs=args.runs,
                            budget=args.budget, workdir=workdir,
                            step_timeout=args.step_timeout, base_env=base_env)
            doc["routes"][route] = res
            _write(args.out, doc)            # what was measured survives a later kill
            print(_line(res, args.budget, args.runs), flush=True)
    finally:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)
    doc["complete"] = True
    doc["passed"] = all(r["passed"] for r in doc["routes"].values())
    _write(args.out, doc)
    print(f"install_smoke: courant-dev {version}: "
          + ("every route reached a verdict on the demo in under "
             f"{args.budget:g} s" if doc["passed"] else "FAILED"))
    return 0 if doc["passed"] else 1


def _fmt(s: float | None) -> str:
    return "n/a" if s is None else f"{s:.1f} s"


def _line(res: dict, budget: float, runs: int) -> str:
    route = res["route"]
    if "error" in res:
        return f"{route:<10} FAILED: {res['error']}"
    passed = sum(r["passed"] for r in res["runs"])
    head = (f"{route:<10} median {_fmt(res['median_s'])}, slowest {_fmt(res['slowest_s'])}, "
            f"{passed} of {runs} runs passed")
    if res["passed"]:
        return f"{head}, under {budget:g} s"
    why = []
    for r in res["runs"]:
        bad = [s for s in r["steps"] if not s["ok"]]
        if bad:
            why.append(f"run {r['run']} {bad[0]['name']}: {bad[0]['why']}")
        elif not r["under_budget"]:
            why.append(f"run {r['run']} took {r['seconds']:.1f} s, over {budget:g} s")
    return f"{head}; " + "; ".join(why)


if __name__ == "__main__":
    sys.exit(main())
