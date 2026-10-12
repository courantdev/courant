# What Courant contacts over the network

The rule (TRD SEC-1): no Courant command opens a network connection unless that is its
documented purpose, you ran it, and it names what it contacts before it does. Courant has no
account, no telemetry and no service of its own: nothing is sent to Courant, ever.

`tests/unit/test_network.py` holds this true. It runs every visible command and every tool of
the MCP server on the demo data with network connections refused, and fails if any of them
tries to connect. The commands below that do contact something are run the same
way: each must name its host first and then stop cleanly, or contact nothing at all when the
call would be refused anyway.

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
