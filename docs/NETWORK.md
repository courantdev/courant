# What Courant contacts over the network

The rule (TRD SEC-1): no Courant command opens a network connection unless that is its
documented purpose, you ran it, and it names what it contacts before it does. Courant has no
account, no telemetry and no service of its own: nothing is sent to Courant, ever.

These tests in this repository hold this true:

- `tests/unit/test_public_network.py` runs `courant verify` on the demo runs that ship inside the
  package, in each of its outputs, with every socket and every new process refused, and fails
  if it makes one. It then runs every visible command and every tool of the MCP server on the
  demo data with network connections refused, and fails if any of them tries to connect. The
  commands below that do contact something are run the same way: each must name its host first
  and then stop cleanly, or contact nothing at all when the call would be refused anyway.
- `tests/unit/test_public_verdict_path.py` reads the code the verdict is made of: the gate, the
  checks it calls, and the commands and tools that read a run, with everything they import.
  No module of it imports a language-model client or a network library, names a model
  provider's endpoint, or starts a process (but for the workers `--jobs` asks for), and
  importing all of it loads no model client. A verdict is neither made by a model nor fetched
  over a network.

## Commands that contact something

| Command | Contacts | When | What is sent |
|---|---|---|---|
| `courant update` | github.com (the releases of `courantdev/courant`, and the download host GitHub redirects to) | A packaged binary, when you run it without `--check`. A pip installation and `--check` contact nothing | It downloads the binary for your platform, the release's `SHA256SUMS` and its signature, and installs nothing unless the signature checks against the release key built in (OPS-7). Nothing about your runs |
| `courant doctor --models` | openrouter.ai | Only with `--models` | Requests for OpenRouter's public model and endpoint listings, without a key. Nothing about your runs |
| `courant test run --judge llm:<provider>:<model>` | openrouter.ai for `openrouter`; api.anthropic.com for `anthropic` (paid: refused unless `COURANT_ALLOW_PAID_LLM=1` is set) | Only with an `llm:` judge, and only once the model door lets that model judge (it must be measured for judging, RRD ARC-12). Until then the command stops with exit 3 and contacts nothing | With your own API key: a request for the key's daily allowance, then each Courant Test task's text |
| `courant spec extract TASK` | The provider the model door chooses: openrouter.ai | Only when the door lets the call through. Until a model is measured for extraction (RRD RM-23), it stops with exit 4 and contacts nothing | The task's text you gave it. With the default `--data customer`, only to an endpoint that retains nothing (ARC-9) |

Each of these prints the host before the first request, for example
`contacting openrouter.ai (openrouter) with your OPENROUTER_API_KEY`. The key's value is never
printed or logged.

An Anthropic key that is not scoped to a workspace needs every request to name one. Set
`ANTHROPIC_WORKSPACE_ID` (in the environment or `.env`, like the key) and each request to
api.anthropic.com carries it as the `anthropic-workspace-id` header. Nothing else is added, and
its value is never printed or logged. Without it, such a key is answered HTTP 400
`invalid_request_error`, and the command stops with
`Anthropic rejected the request (HTTP 400, invalid_request_error): the key is not scoped to a workspace ...`:
a request the provider rejected is never scored as the model's answer (TRD ARC-19).


## Commands that contact nothing

`courant verify` (including `--batch`, `--pack`, `--check-pack`, `--record`, `--about`),
`courant gate`, `courant spec init`, `hash`, `check`, `confirm` and `lock`, `courant
convergence`, `courant preflight`, `courant audit`, `courant watch`, `courant test build`,
`courant test run` with the `courant` or `constant:` judges, `courant demo`, `courant mcp` and
`courant-mcp` (the server talks to its client over stdin and stdout only), `courant doctor`
(and `--bundle`, which writes a file for you to send yourself), and `courant version`.

`courant doctor` asks the local Docker daemon whether it is running, over the daemon's own
socket, or the address in `DOCKER_HOST` if you set one. That is your machine's daemon, not a
Courant service.


## Outside the commands

- The installer, `curl -fsSL https://github.com/courantdev/courant/releases/latest/download/install.sh | sh`, downloads the
  binary, `SHA256SUMS` and its signature from the releases of github.com/courantdev/courant, and
  checks the signature when python3 with `cryptography` or OpenSSL 3 is there (it says when
  neither is, and that only the checksum was checked). The GitHub Action
  runs the same installer from its own repository; the check it then runs contacts nothing.
- Installing with pip downloads the `courant-dev` package from the package index you use.
  PyPI's `courant` is an unrelated project: install `courant-dev`. The command is `courant`.

## For a privacy page

> Courant runs on your machine and has no account, no telemetry and no service of its own; it
> never sends your runs, cases or results anywhere. Its checks open no network connection. Only
> these contact the network, only when you run them, and each names the host first:
> `courant update` downloads a release from github.com; `courant doctor --models` reads
> OpenRouter's public model list; and a language model you choose (an `llm:` judge for the
> Courant Test, or `courant spec extract`) sends that task's text to the model provider with
> your own API key.
