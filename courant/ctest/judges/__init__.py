"""Judges (TRD CT-5): anything that answers a Courant Test task.

    courant               the verifier itself, through the product entry point
    constant:<VERDICT>    a baseline that always says the same thing
    llm:<provider>:<m>    a language model, given the fixed prompt judge/1
"""

from __future__ import annotations


def make(spec: str, client=None, measurement: str | None = None, door=None):
    """A judge from its spec. ``measurement`` and ``door`` reach an ``llm:`` judge's model
    door: the pre-registered run its calls belong to (ARC-12), and the door to call through."""
    from courant.ctest.judges.constant import ConstantJudge
    from courant.ctest.judges.courant import CourantJudge

    if spec == "courant":
        return CourantJudge()
    if spec.startswith("constant:"):
        return ConstantJudge(spec.split(":", 1)[1])
    if spec.startswith("llm:"):
        from courant.ctest.judges.llm import LLMJudge

        _, provider, model = spec.split(":", 2)
        return LLMJudge(provider=provider, model=model, client=client,
                        measurement=measurement, door=door)
    raise ValueError(f"unknown judge {spec!r}: use courant, constant:<VERDICT> or llm:<provider>:<model>")
