"""No language model, and no network, in the verdict's path (TRD INV-1, INV-5; README, "The rule
this repository keeps"; docs/NETWORK.md).

Deterministic code decides every verdict, number and grade. So nothing the verdict is made of
may be, or may reach, a language model, and nothing the checks do may use the network. This
file reads that off the syntax tree of the package, so it runs the same here and in the public
repository, and needs no private code.

The verdict's path is the gate (`courant.verify`), the commands and tools that hand its verdict
to a person, a CI system or an agent, and everything they import, transitively, wherever the
import statement sits (inside a function or under `if TYPE_CHECKING:` included) and with the
parent packages that run first. Over that path:

1. no module imports a language-model client: the model door (`courant.llm`) or a vendor's SDK;
2. no module names a model provider's endpoint in code;
3. no module imports a network library;
4. no module starts a process, except the worker pool `--jobs` asks for;
5. and a fresh interpreter that imports all of it loads no model client.

The lint is read from syntax, not from text, so a comment or a docstring cannot trip it and an
aliased import cannot hide from it. Each rule is first run on snippets that break it, so the lint
is known to see what it guards. The model door itself is found to be imported by commands outside
the path, so the scan is known not to be passing on an empty result.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import courant

PKG = Path(courant.__file__).resolve().parent
ROOT = PKG.parent

# ---------------------------------------------------------------------------- what is guarded

#: A language-model client: the model door, the vendors' SDKs and the frameworks that wrap them.
MODEL_CLIENTS = (
    "courant.llm", "openai", "anthropic", "google.generativeai", "google.genai", "cohere",
    "mistralai", "groq", "together", "replicate", "litellm", "ollama", "boto3", "langchain",
    "langchain_core", "langchain_openai", "langchain_anthropic", "llama_index", "llama_cpp",
    "transformers", "huggingface_hub", "vllm",
)
#: What a model call travels over.
NETWORK_LIBRARIES = (
    "socket", "ssl", "socketserver", "http.client", "http.server", "urllib.request", "ftplib",
    "smtplib", "poplib", "imaplib", "telnetlib", "xmlrpc", "httpx", "httpcore", "requests",
    "urllib3", "aiohttp", "h11", "websocket", "websockets", "docker",
)
#: What starts a process. (`os.system(...)` and the rest of its family are found as calls.)
PROCESS_LIBRARIES = (
    "subprocess", "multiprocessing", "concurrent.futures.process",
    "concurrent.futures.ProcessPoolExecutor", "pty", "pexpect", "os.system", "os.popen",
    "os.fork", "os.forkpty", "os.posix_spawn", "os.posix_spawnp",
)
PROCESS_CALLS = {"system", "popen", "fork", "forkpty", "posix_spawn", "posix_spawnp",
                 "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe",
                 "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp",
                 "spawnvpe"}
#: A model provider's endpoint, as a string a module would build a request from.
MODEL_ENDPOINTS = ("openrouter.ai", "api.anthropic.com", "api.openai.com",
                   "generativelanguage.googleapis.com", "api.cohere.ai", "api.mistral.ai",
                   "api.groq.com", "chat/completions", "messages.create")
#: The one place a verdict-path module starts processes: `courant verify --batch --jobs N` runs
#: this same verifier in a pool of worker processes. It is opt-in (the default is one), and what
#: the workers run is the verifier, never a solver.
PROCESS_ALLOWED = {"courant.verify.batch": ("multiprocessing", "concurrent.futures")}

#: Packages whose every module hands a verdict on, or decides one.
VERDICT_PACKAGES = ("verify", "gate", "mcp", "record", "live")
#: Commands that read a run and print what they found.
VERDICT_COMMANDS = ("verify", "gate", "tour", "live", "mcp", "preflight", "convergence", "audit")
#: Commands whose purpose is to reach a model or the network, and which say so (NETWORK.md).
MAY_REACH_OUT = ("ctest", "spec", "doctor", "update")


# ---------------------------------------------------------------------------- reading imports


def _is_or_under(name: str, module: str) -> bool:
    return name == module or name.startswith(module + ".")


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imports_of(src: str, module: str, *, is_package: bool = False) -> list[tuple[int, str]]:
    """(line, dotted name) of every import in `src`, a module named `module`, relative imports
    resolved. `from courant.llm import openrouter` gives both `courant.llm` and
    `courant.llm.openrouter`, so a submodule imported by name is seen. A string given to
    `importlib.import_module` or `__import__` counts as an import."""
    package = module.split(".") if is_package else module.split(".")[:-1]
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            found += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                name = ".".join(base + ([node.module] if node.module else []))
            else:
                name = node.module or ""
            found.append((node.lineno, name))
            found += [(node.lineno, f"{name}.{a.name}") for a in node.names]
        elif isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if (called in ("import_module", "__import__") and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                found.append((node.lineno, node.args[0].value))
    return found


def _code_strings(src: str) -> list[tuple[int, str]]:
    """(line, text) of every string literal in the code, docstrings left out."""
    tree = ast.parse(src)
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docs.add(id(body[0].value))
    return [(n.lineno, n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


def reaches(src: str, module: str, libraries: tuple[str, ...], *,
            is_package: bool = False) -> list[str]:
    """Why `src` imports one of `libraries`, or [] (each reason names the line)."""
    return [f"line {line}: imports {name}"
            for line, name in imports_of(src, module, is_package=is_package)
            if any(_is_or_under(name, lib) for lib in libraries)]


def calls_a_process(src: str) -> list[str]:
    """Each `os.system(...)`-style call in `src`, by line."""
    return [f"line {n.lineno}: calls os.{n.func.attr}()"
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in PROCESS_CALLS
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "os"]


def names_an_endpoint(src: str) -> list[str]:
    """Each string in the code of `src` that names a model provider's endpoint."""
    return [f"line {line}: names {needle!r}"
            for line, text in _code_strings(src)
            for needle in MODEL_ENDPOINTS if needle in text]


# ---------------------------------------------------------------------------- the path


def _file_of(name: str) -> Path | None:
    base = ROOT.joinpath(*name.split("."))
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    return None


def _modules_of(directory: str) -> list[str]:
    return [_module_name(p) for p in sorted((PKG / directory).glob("*.py"))]


def roots() -> list[str]:
    out = []
    for pkg in VERDICT_PACKAGES:
        out += _modules_of(pkg)
    out += [f"courant.cli.commands.{c}" for c in VERDICT_COMMANDS]
    return out


def verdict_path() -> dict[str, Path]:
    """{module: file} of the verdict's path: the roots and everything they import, transitively.
    A model client is not followed into: it is reported where it is imported."""
    seen: dict[str, Path] = {}
    todo = roots()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        path = _file_of(name)
        if path is None:                       # a name inside a module, or a third-party module
            continue
        seen[name] = path
        parts = name.split(".")
        todo += [".".join(parts[:k]) for k in range(1, len(parts))]       # the parents run first
        for _, imported in imports_of(path.read_text(), name, is_package=path.name == "__init__.py"):
            if imported.split(".")[0] == "courant" and not any(
                    _is_or_under(imported, c) for c in MODEL_CLIENTS):
                todo.append(imported)
    return dict(sorted(seen.items()))


def _scan(rule) -> dict[str, list[str]]:
    """{module: reasons} of the verdict path, for the modules `rule(module, source)` reports."""
    out = {}
    for name, path in verdict_path().items():
        found = rule(name, path.read_text(), path.name == "__init__.py")
        if found:
            out[name] = found
    return out


# ---------------------------------------------------------------------------- the lint sees

SNIPPETS_MODEL = {
    "import openai": "imports openai",
    "import anthropic as a": "imports anthropic",
    "from anthropic import Anthropic": "imports anthropic",
    "from openai.types import chat": "imports openai.types",
    "from courant.llm import default_door": "imports courant.llm",
    "from courant import llm": "imports courant.llm",
    "from courant.llm.openrouter import send": "imports courant.llm.openrouter",
    "from google import genai": "imports google.genai",
    "import litellm": "imports litellm",
    "def f():\n    import openai\n": "imports openai",
    "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from anthropic import Client\n":
        "imports anthropic",
    "import importlib\nm = importlib.import_module('openai')": "imports openai",
    "m = __import__('anthropic')": "imports anthropic",
}


def test_the_lint_sees_each_way_to_reach_a_model():
    for src, why in SNIPPETS_MODEL.items():
        assert any(why in r for r in reaches(src, "courant.verify.x", MODEL_CLIENTS)), src
    # a relative import is resolved, from a module and from a package
    assert reaches("from ..llm import door", "courant.verify.x", MODEL_CLIENTS)
    assert reaches("from ..llm import door", "courant.verify", MODEL_CLIENTS, is_package=True)
    assert reaches("from .llm import door", "courant.x", MODEL_CLIENTS)
    assert not reaches("from .llm import door", "courant.verify.x", MODEL_CLIENTS)  # verify.llm
    ok = ('"""Never imports openai, anthropic or courant.llm."""\n'
          "# import openai\nimport json\nfrom courant.verify import verify_path\n"
          "from courant.verification import grade\n")
    assert reaches(ok, "courant.verify.x", MODEL_CLIENTS) == []


def test_the_lint_sees_a_network_library_a_process_and_an_endpoint():
    for src, why in {
        "import httpx": "imports httpx",
        "import requests": "imports requests",
        "from urllib.request import urlopen": "imports urllib.request",
        "from urllib import request": "imports urllib.request",
        "import http.client": "imports http.client",
        "import socket": "imports socket",
        "from http import client": "imports http.client",
        "def f():\n    import aiohttp\n": "imports aiohttp",
    }.items():
        assert any(why in r for r in reaches(src, "courant.verify.x", NETWORK_LIBRARIES)), src
    for src, why in {
        "import subprocess": "imports subprocess",
        "from subprocess import run": "imports subprocess",
        "import multiprocessing": "imports multiprocessing",
        "from concurrent.futures import ProcessPoolExecutor": "ProcessPoolExecutor",
        "from os import system": "imports os.system",
    }.items():
        assert any(why in r for r in reaches(src, "courant.verify.x", PROCESS_LIBRARIES)), src
    for src in ("import os\nos.system('ls')", "import os\nos.popen('ls')", "os.fork()",
                "import os\nos.execv('/bin/ls', ['ls'])", "import os\nos.spawnl(0, 'ls')"):
        assert calls_a_process(src), src
    assert calls_a_process("import os\nos.path.join('a', 'b')\nsystem = 1\n") == []
    for src in ("URL = 'https://openrouter.ai/api/v1/chat/completions'",
                "HOST = 'api.anthropic.com'", "CALL = 'client.messages.create'"):
        assert names_an_endpoint(src), src
    docstring_and_comment = ('"""Posts nothing to openrouter.ai."""\n'
                             "# api.anthropic.com is only a comment here\n"
                             "def f():\n    '''chat/completions, in a docstring'''\n    return 1\n")
    assert names_an_endpoint(docstring_and_comment) == []


# ---------------------------------------------------------------------------- the path is found


def test_the_verdict_path_is_found_and_is_not_empty():
    path = verdict_path()
    must_be_on_it = ["courant.verify", "courant.verify.pipeline", "courant.verify.policy",
                     "courant.verify.ingest", "courant.verify.batch", "courant.gate.run",
                     "courant.mcp.tools", "courant.record.record", "courant.live.watch",
                     "courant.cli.commands.verify", "courant.verification",
                     "courant.converge", "courant.spec.check", "courant.watcher",
                     "courant.case", "courant.events"]
    missing = [m for m in must_be_on_it if m not in path]
    assert missing == [], f"the scan does not reach {missing}: it would guard too little"
    assert len(path) >= 40, f"the scan found only {len(path)} modules"
    assert all(p.is_file() for p in path.values())
    for r in roots():
        assert _file_of(r) is not None, f"{r} is named as a root and does not exist"


def test_the_path_reaches_through_a_function_and_a_parent_package(tmp_path, monkeypatch):
    """An import inside a function is followed, and so is the `__init__` of the package it is in:
    Python runs it first, so a model client imported there is on the path."""
    here = sys.modules[__name__]
    (tmp_path / "courant" / "verify").mkdir(parents=True)
    (tmp_path / "courant" / "deep").mkdir()
    (tmp_path / "courant" / "__init__.py").write_text("")
    (tmp_path / "courant" / "verify" / "__init__.py").write_text("")
    (tmp_path / "courant" / "verify" / "a.py").write_text(
        "def f():\n    import courant.deep.b\n")
    (tmp_path / "courant" / "deep" / "__init__.py").write_text("import courant.parent_side\n")
    (tmp_path / "courant" / "deep" / "b.py").write_text("X = 1\n")
    (tmp_path / "courant" / "parent_side.py").write_text("import openai\n")
    monkeypatch.setattr(here, "ROOT", tmp_path)
    monkeypatch.setattr(here, "PKG", tmp_path / "courant")
    monkeypatch.setattr(here, "VERDICT_PACKAGES", ("verify",))
    monkeypatch.setattr(here, "VERDICT_COMMANDS", ())
    path = verdict_path()
    assert {"courant.verify.a", "courant.deep", "courant.deep.b", "courant.parent_side"} <= set(path)
    found = _scan(lambda name, src, pkg: reaches(src, name, MODEL_CLIENTS, is_package=pkg))
    assert list(found) == ["courant.parent_side"]


def test_the_model_door_is_found_outside_the_path():
    """The scan can find the door: modules outside the verdict's path import it, and none of them
    is on the path. (Otherwise a clean result could mean the scan sees nothing.)"""
    assert (PKG / "llm" / "__init__.py").is_file(), "the model door is not where this expects it"
    path = verdict_path()
    users = []
    for p in sorted(PKG.rglob("*.py")):
        if "__pycache__" in p.parts or "llm" in p.relative_to(PKG).parts[:1]:
            continue
        name = _module_name(p)
        if reaches(p.read_text(), name, ("courant.llm",), is_package=p.name == "__init__.py"):
            users.append(name)
    assert users, "no module imports the model door: the scan is not looking at the right thing"
    assert [u for u in users if u in path] == [], "a module on the verdict's path imports the door"


def test_every_command_module_is_on_one_side():
    """A new command is either read for the verdict's path or says it reaches out, never neither."""
    commands = {p.stem for p in (PKG / "cli" / "commands").glob("*.py")} - {"__init__"}
    assert commands == set(VERDICT_COMMANDS) | set(MAY_REACH_OUT), (
        "classify the new command: VERDICT_COMMANDS (it reads a run) or MAY_REACH_OUT (its "
        f"purpose is a model or the network, and docs/NETWORK.md says so): {sorted(commands)}")
    assert not set(VERDICT_COMMANDS) & set(MAY_REACH_OUT)


# ---------------------------------------------------------------------------- the path is clean


def test_no_module_in_the_verdict_path_imports_a_language_model_client():
    """INV-1: a model client is imported only where a model is called on purpose, never by the
    gate, the checks it calls, or the commands and tools that print its verdict."""
    found = _scan(lambda name, src, pkg: reaches(src, name, MODEL_CLIENTS, is_package=pkg))
    assert found == {}, f"a language-model client is on the verdict's path: {found}"


def test_no_module_in_the_verdict_path_names_a_model_endpoint():
    found = _scan(lambda name, src, pkg: names_an_endpoint(src))
    assert found == {}, f"a model provider's endpoint is named on the verdict's path: {found}"


def test_no_module_in_the_verdict_path_imports_a_network_library():
    """The checks never use the network (docs/NETWORK.md)."""
    found = _scan(lambda name, src, pkg: reaches(src, name, NETWORK_LIBRARIES, is_package=pkg))
    assert found == {}, f"a network library is on the verdict's path: {found}"


def test_no_module_in_the_verdict_path_starts_a_process():
    """Courant never runs a solver: the verdict's path starts no process, except the pool of
    workers `--jobs` asks for, which runs the verifier itself."""
    def rule(name, src, pkg):
        allowed = PROCESS_ALLOWED.get(name, ())
        return [r for r in reaches(src, name, PROCESS_LIBRARIES, is_package=pkg)
                + calls_a_process(src)
                if not any(a in r for a in allowed)]

    found = _scan(rule)
    assert found == {}, f"a process is started on the verdict's path: {found}"
    # the allowance is used: if batch.py stops needing it, it goes
    batch = (PKG / "verify" / "batch.py").read_text()
    assert reaches(batch, "courant.verify.batch", PROCESS_LIBRARIES), \
        "PROCESS_ALLOWED names a module that no longer starts processes: remove it"


# ---------------------------------------------------------------------------- at run time


def test_importing_the_verdict_path_loads_no_model_client():
    """And not transitively, whatever a third-party package imports: import every module of the
    path in a fresh interpreter and look at what is loaded."""
    probe = textwrap.dedent("""
        import importlib, json, sys
        for name in json.loads(sys.argv[1]):
            try:
                importlib.import_module(name)
            except ModuleNotFoundError as exc:         # an optional extra that is not installed
                if (exc.name or "").split(".")[0] == "courant":
                    raise
        clients = json.loads(sys.argv[2])
        loaded = sorted(m for m in sys.modules
                        if any(m == c or m.startswith(c + ".") for c in clients))
        print(json.dumps(loaded))
    """)
    env = {**os.environ, "PYTHONPATH": str(ROOT), "COURANT_NO_DOTENV": "1"}
    out = subprocess.run([sys.executable, "-c", probe, json.dumps(sorted(verdict_path())),
                          json.dumps(list(MODEL_CLIENTS))],
                         cwd=ROOT, env=env, check=False, capture_output=True, text=True,
                         timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    loaded = json.loads(out.stdout.strip().splitlines()[-1])
    assert loaded == [], f"importing the verdict's path loads a model client: {loaded}"
