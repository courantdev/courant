"""courant-mcp: the verifier as tools for AI agents (TRD §8).

An agent that has just run a simulation calls `verify_run` before it reports
success. It gets the same typed verdict `courant verify --json` prints, with
the reasons and the suggested next action. That is how an agent learns to
"identify when it's wrong and pause", which engineers asked for at NAFEMS 2026.
"""
