"""OpenFOAM's keyword patterns (`wordRe`), matched with `re`'s meaning and a bounded amount of work.

A quoted keyword in an OpenFOAM dictionary is a regular expression that must match a whole name:
`"(inlet|outlet)"` in a `boundaryField`, `".*"` under `relaxationFactors`, `"(U|k)Final"` under
`solvers`. The readers matched it with Python's `re`, a backtracking engine, against a name the
specification or the log supplied, so the case decided how long the match took: on `"(.*)*!"`
the time doubled with each character of the patch name, and one such line in `0/U` stalled
`courant verify --spec` with no output (hard-proofing audit, 21 Oct 2026).

Here the pattern runs as a Thompson automaton (a Pike VM) over Python's own parse of it. Every
test of one character or of one position that is not a plain comparison (a literal under
IGNORECASE, a class, an anchor) is a one-item pattern compiled by `re`'s own compiler, so it
means what `re` means, case folding and categories included, and one character cannot backtrack.
The rest is this module's: concatenation, alternation and repetition, whose language does not
depend on the order `re` tries them in. Evidence, committed: `tests/unit/test_wordre.py`, 80,000
seeded pattern-name pairs over every construct evaluated and 8,736 nested- and empty-loop pairs,
0 differences from `re.fullmatch` (the review of 21 Oct 2026 found the folding of `İ` and of a
class range that this module had written itself differ).

A pattern that is a set of names, plain characters with an alternation of plain-character
sequences after them inside any enclosing groups (a boundaryField's `"(wing_upper|wing_lower|...)"`)
and no IGNORECASE, is matched by membership instead (`_NameSet`): the same answer, `re.fullmatch`'s,
at a cost of the name's length however many names it holds, where its automaton passed
`MAX_STATES` at 42 names of ten characters (the review of the 0.6.0 hard-proofing, 22 Oct 2026).
Every other pattern keeps the automaton. Evidence: of 1,000 seeded alternations of names (20,000
pattern-name pairs), 637 patterns are matched as sets (12,740 pairs; 59 of the 637 are past
`MAX_STATES`), with 0 differences from `re.fullmatch`, and none from the automaton where it
answers.

`csearch` is OpenFOAM v2406's `dictionary::csearch` on these matches, the one lookup of a block or
an entry by name: a key whose text is the name first, quoted or not, then the patterns from the
last declared.

The work is bounded three ways, so a case cannot stall the gate however many keywords it writes:
- an automaton has at most `MAX_STATES` states (a bounded repeat is unrolled: `a{3}` is three),
  and a character class spans at most `MAX_CLASS_SPAN` code points;
- every match is charged what it may cost (compiling the pattern once, then its states times the
  name's length plus one) against a budget: `LOOKUP_STEPS` for one dictionary lookup
  (`first_match`, which tries every keyword against one name), `RUN_STEPS` for a whole verify or
  preflight (`budget`), `MATCH_STEPS` for a match made alone;
- what the budget cannot pay for is not evaluated.
A pattern that needs backtracking to mean what it says (a backreference, a lookaround, a
possessive or atomic group, a conditional), an automaton past `MAX_STATES` or a class past
`MAX_CLASS_SPAN`, or a match the budget cannot pay for, is not evaluated: `fullmatch` returns None, and the caller says the
keyword could not be read rather than guess.

Deterministic: the cost charged is a function of the pattern and the name, and of the matches
made before it under the same budget, never of a cache or the clock. No I/O.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, wraps

try:                                                     # Python 3.11+
    from re import _compiler as _sre_compile
    from re import _constants as _c
    from re import _parser as _sre_parse
except ImportError:                                      # pragma: no cover - Python < 3.11
    import sre_compile as _sre_compile                   # type: ignore[no-redef]
    import sre_constants as _c                           # type: ignore[no-redef]
    import sre_parse as _sre_parse                       # type: ignore[no-redef]

#: The largest automaton evaluated. OpenFOAM's own patterns need tens of states (32 is the
#: largest in the committed cases, `"(U|k|omega|gammaInt|ReThetat)"`); 4,803 were allowed before
#: the review of 21 Oct 2026, so one keyword cost some 5 ms a name.
MAX_STATES = 500
#: The most code points a character class's ranges may span. `re`'s compiler visits each one
#: (`[\x00-￿]` is 65,536 visits, milliseconds), and its cost would not be the pattern's
#: length; OpenFOAM's patterns use `[0-9]` and `[A-Za-z]`.
MAX_CLASS_SPAN = 256
#: What compiling a pattern is charged, before it is compiled: per character (parsing it, and
#: one `re` compilation, about 4 µs, per distinct one-character test) and per state it may build.
#: A step is one automaton state visited, about 0.1 µs.
COMPILE_STEPS_PER_CHAR = 50
COMPILE_STEPS_PER_STATE = 10
#: One dictionary lookup: every keyword tried against one name.
LOOKUP_STEPS = 1_000_000
#: A whole verify or preflight (`budget`): every lookup it makes, together.
RUN_STEPS = 20_000_000
#: A match made outside any lookup or verify.
MATCH_STEPS = 2_000_000
#: The longest pattern one lookup can pay to compile (`_answer`'s charge for a pattern it has not
#: compiled: `COMPILE_STEPS_PER_CHAR` a character and `COMPILE_STEPS_PER_STATE` a state it may
#: build), 19,900 characters: no longer pattern is ever parsed here. `literal_prefix` parses none
#: past it either: `re`'s parser moves an alternation's common start out of it one character a
#: round (re/_parser.py, `_parse_sub`'s `del item[0]` on every alternative), work that grows with
#: the square of the shared length, so a case-written key of two alternatives sharing a megabyte
#: took minutes (the review of hp6/integrate).
MAX_PARSED_CHARS = (LOOKUP_STEPS - COMPILE_STEPS_PER_STATE * MAX_STATES) // COMPILE_STEPS_PER_CHAR

_METACHARS = frozenset(".^$*+?{}[]\\|()")
_TYPE_FLAGS = _sre_parse.TYPE_FLAGS


class _Unsupported(Exception):
    """A construct the automaton cannot express, or one too large to evaluate."""


# ---------------------------------------------------------------- the budget
class _Budget:
    """The steps left, the automata already paid for under this budget, and the answers given
    under it: one keyword asked of one name twice in a verify gets one answer, so two checks
    that make the same lookup agree, whatever the budget did in between."""

    __slots__ = ("answers", "left", "programs")

    def __init__(self, steps: int) -> None:
        self.left = steps
        self.programs: dict[str, tuple | bool | None] = {}
        self.answers: dict[tuple[str, str], bool | None] = {}


#: The most answers one budget remembers (some ten megabytes); past it an answer is still given,
#: and paid for, but not remembered.
MAX_ANSWERS = 100_000
_MISSING = object()


_BUDGETS: ContextVar[tuple[_Budget, ...]] = ContextVar("courant_wordre_budgets", default=())


@contextmanager
def budget(steps: int = RUN_STEPS):
    """Bound every match made inside to `steps` in all: a verify, a preflight. A budget opened
    inside another draws on both; an automaton is paid for once under the outermost."""
    token = _BUDGETS.set((*_BUDGETS.get(), _Budget(steps)))
    try:
        yield
    finally:
        _BUDGETS.reset(token)


def bounded(fn):
    """`fn`, every match it makes drawing on one `budget`: a verify, a preflight, a conformance
    check, a draft."""
    @wraps(fn)
    def run(*args, **kwargs):
        with budget():
            return fn(*args, **kwargs)
    return run


def _pay(budgets: tuple[_Budget, ...], cost: int) -> bool:
    if any(b.left < cost for b in budgets):
        return False
    for b in budgets:
        b.left -= cost
    return True


# ---------------------------------------------------------------- matching
def fullmatch(pattern: str, name: str) -> bool | None:
    """Whether `name` matches `pattern` in full, as `re.fullmatch(pattern, name)` says.

    False for a pattern `re` refuses, as the readers have always taken one (no keyword matches
    it). None when the answer is not given: a construct that needs backtracking, an automaton
    over `MAX_STATES`, or a match its budget cannot pay for.
    """
    if not _METACHARS.intersection(pattern):
        return pattern == name                           # a literal keyword: no automaton
    if not _BUDGETS.get():
        with budget(MATCH_STEPS):
            return _charged(pattern, name)
    return _charged(pattern, name)


def _charged(pattern: str, name: str) -> bool | None:
    budgets = _BUDGETS.get()
    outermost = budgets[0]
    known = outermost.answers.get((pattern, name), _MISSING)
    if known is not _MISSING:
        return known
    answer = _answer(budgets, pattern, name)
    if len(outermost.answers) < MAX_ANSWERS:
        outermost.answers[(pattern, name)] = answer
    return answer


def _answer(budgets: tuple[_Budget, ...], pattern: str, name: str) -> bool | None:
    programs = budgets[0].programs
    if pattern in programs:
        prog = programs[pattern]
    else:
        cost = COMPILE_STEPS_PER_CHAR * len(pattern) + COMPILE_STEPS_PER_STATE * MAX_STATES
        if not _pay(budgets, cost):
            return None
        prog = programs[pattern] = _compile(pattern)
    if prog is False or prog is None:
        return prog
    if isinstance(prog, _NameSet):                       # a set: the name's length, no states
        return prog.fullmatch(name) if _pay(budgets, len(name) + 1) else None
    if not _pay(budgets, len(prog) * (len(name) + 1)):
        return None
    return _run(prog, name)


def first_match(keys: Sequence[str], name: str, *,
                pattern: Callable[[str], str | None] = lambda key: key,
                ) -> tuple[int | None, tuple[int, ...]]:
    """One dictionary lookup: the index of the first of `keys` that names `name` (None when none
    does), and the indices of the keys before it that were not evaluated, any of which may be the
    one OpenFOAM finds. The caller gives the keys in the order OpenFOAM tries them, and treats
    a lookup with such a key before its match as one it could not make.

    `pattern(key)` is the regular expression a key is, or None for a key that is a name and is
    compared as one (every key is a pattern by default). Bounded as a whole: `LOOKUP_STEPS`, and
    any budget around it."""
    unsure: list[int] = []
    with budget(LOOKUP_STEPS):
        for i, key in enumerate(keys):
            regex = pattern(key)
            hit = key == name if regex is None else fullmatch(regex, name)
            if hit:
                return i, tuple(unsure)
            if hit is None:
                unsure.append(i)
    return None, tuple(unsure)


def csearch(keys: Sequence[tuple[str, bool]], name: str, *, dictionaries: bool = True
            ) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """OpenFOAM v2406's `dictionary::csearch` of `name` among a dictionary's `keys`, each (its
    text, whether it is a pattern: a quoted keyword), in declaration order. The one lookup of a
    block or an entry by name (`deck.block`, `convergence.equation_relaxation`).

    Keys that share a text are one entry, as v2406 reads a dictionary (`entry::New` under the
    default `#inputMode merge` finds the earlier one by its text and `dictionary::add`s the later
    to it), and the entry keeps the first's keyword, so it is a pattern when the first was
    quoted. Its place among the patterns (`patterns_`, which `findInPatterns` walks from the
    front, the last added first) depends on what the entries are. `dictionaries` (blocks,
    `deck.block`): the later is merged into the first, which keeps its place, so `"in.*" {..}
    ".*" {..} "in.*" {..}` finds `".*"` for `inlet1` (the review of the 0.6.0 follow-ups found
    the later copy put first and its body returned alone). Otherwise (values, the relaxation
    factors): the later replaces the entry and is pushed to the front, the place of its last
    writing.

    The hash table first, which holds a keyword by its text: every key whose text is `name`,
    quoted or not (`"U"` is found by `U`, before any pattern). Otherwise the patterns, from the
    front, and the first of them that matches `name` in full. Returns (the indices of the entry
    found, every key it was written under, in order: merged in that order for blocks, the last
    standing for values; or none; the indices of the patterns tried before the one found, or
    before the end, that were not evaluated, any of which may be the one OpenFOAM finds). Bounded
    as `first_match` is."""
    exact = tuple(i for i, (text, _) in enumerate(keys) if text == name)
    if exact:
        return exact, ()
    written: dict[str, list[int]] = {}
    for i, (text, _) in enumerate(keys):
        written.setdefault(text, []).append(i)
    place = 0 if dictionaries else -1
    entries = sorted((at for at in written.values() if keys[at[0]][1]),
                     key=lambda at: at[place], reverse=True)
    hit, unsure = first_match([keys[at[0]][0] for at in entries], name)
    return (tuple(entries[hit]) if hit is not None else ()), \
        tuple(i for j in unsure for i in entries[j])


@lru_cache(maxsize=1024)
def literal_prefix(pattern: str) -> str:
    """The characters every name `pattern` matches in full begins with, as `re` parses it: its
    plain characters up to the first that is not one, inside any groups, and after them the start
    the alternatives of an alternation share. "" when `re` refuses the pattern or this cannot
    read it, and no character under IGNORECASE (a folded character is more than one). Never
    longer than what a match must begin with, so a caller that asks whether the matches all begin
    with a word can be told no where they do, never yes where they do not.

    `re`'s parser has already moved an alternation's common start out of it and read a class of
    one character as the character (`(pcorr|pcorrFinal)` is `pcorr` then `(|Final)`, `[p]` is
    `p`), so the answer is the pattern's meaning, not its spelling. Deterministic, no I/O; the
    work is the parse, of a pattern of at most `MAX_PARSED_CHARS` characters, which the matcher
    would charge a lookup's budget for; a longer one is not parsed, and its answer is ""."""
    if len(pattern) > MAX_PARSED_CHARS:
        return ""
    try:
        parsed = _sre_parse.parse(pattern, 0)
        return _prefix(_data(parsed), parsed.state.flags)[0]
    except (re.error, RecursionError, OverflowError, ValueError):
        return ""


def _prefix(items, flags: int) -> tuple[str, bool]:
    """(the plain characters `items` begins with, whether they are all of it)."""
    head: list[str] = []
    for op, av in items:
        if op is _c.LITERAL and not flags & re.IGNORECASE:
            head.append(chr(av))
        elif op is _c.SUBPATTERN:
            _group, add, remove, body = av
            inner, whole = _prefix(_data(body), _scoped(flags, add, remove))
            head.append(inner)
            if not whole:
                return "".join(head), False
        elif op is _c.BRANCH:
            starts = [_prefix(_data(alternative), flags)[0] for alternative in av[1]]
            shared = starts[0] if starts else ""
            for start in starts[1:]:
                while not start.startswith(shared):
                    shared = shared[:-1]
            head.append(shared)
            return "".join(head), False
        else:
            return "".join(head), False
    return "".join(head), True


class _NameSet:
    """A pattern that is a set of names: a prefix and the rest of each name, plain characters
    only. Matched by membership, which is `re.fullmatch`'s answer on it, at a cost of the name's
    length whatever the number of names."""

    __slots__ = ("prefix", "rest")

    def __init__(self, prefix: str, rest: frozenset[str]) -> None:
        self.prefix, self.rest = prefix, rest

    def fullmatch(self, name: str) -> bool:
        return name.startswith(self.prefix) and name[len(self.prefix):] in self.rest


def _name_set(parsed, flags: int) -> _NameSet | None:
    """`parsed` as a set of names, when it is one: inside any enclosing groups, plain characters
    and then, optionally, an alternation of plain-character sequences. `re`'s parser moves the
    alternatives' common prefix out (`(wing_upper|wing_lower)` is `wing_` then `(upper|lower)`),
    and writes alternatives of one character each as a class of them (`(wing_a|wing_b)` is
    `wing_[ab]`), so a patch group is this shape. Not under IGNORECASE (`re` folds case by its
    own tables, which the automaton's one-character tests use): None, and the automaton decides."""
    items = list(_data(parsed))
    while len(items) == 1 and items[0][0] is _c.SUBPATTERN:
        _group, add, remove, body = items[0][1]
        flags = _scoped(flags, add, remove)
        items = list(_data(body))
    if flags & re.IGNORECASE:
        return None
    head = []
    for i, (op, av) in enumerate(items):
        last = i == len(items) - 1
        if op is _c.LITERAL:
            head.append(chr(av))
        elif op is _c.BRANCH and last:
            rest = []
            for alternative in av[1]:
                pieces = list(_data(alternative))
                if any(o is not _c.LITERAL for o, _ in pieces):
                    return None
                rest.append("".join(chr(a) for _, a in pieces))
            return _NameSet("".join(head), frozenset(rest))
        elif op is _c.IN and last and all(o is _c.LITERAL for o, _ in av):
            return _NameSet("".join(head), frozenset(chr(a) for _, a in av))
        else:
            return None
    return _NameSet("".join(head), frozenset(("",)))


@lru_cache(maxsize=1024)
def _compile(pattern: str):
    """What `pattern` is matched by: a set of names (`_NameSet`), or the automaton, a tuple of
    instructions; False when `re` refuses the pattern, None when it is not one this matcher
    evaluates."""
    try:
        parsed = _sre_parse.parse(pattern, 0)
    except re.error:
        return False
    except (RecursionError, OverflowError, ValueError):
        return None
    names = _name_set(parsed, parsed.state.flags)
    if names is not None:
        return names
    prog: list = []
    try:
        _emit(parsed, parsed.state.flags, prog)
    except (_Unsupported, RecursionError):
        return None
    except Exception:                                    # noqa: BLE001 - `re` refused a piece
        return None
    prog.append(("match",))
    return tuple(prog)


def _data(sub):
    return sub.data if hasattr(sub, "data") else sub


def _scoped(flags: int, add: int, remove: int) -> int:
    """The flags inside `(?add-remove:...)`, as `re`'s compiler combines them: a type flag
    (ASCII, UNICODE) replaces the one in force."""
    if add & _TYPE_FLAGS:
        flags &= ~_TYPE_FLAGS
    return (flags | add) & ~remove


def _emit(sub, flags: int, prog: list) -> None:
    for op, av in _data(sub):
        if (op is _c.LITERAL or op is _c.NOT_LITERAL) and not flags & re.IGNORECASE:
            c = chr(av)
            prog.append(("char", (lambda ch, c=c: ch == c) if op is _c.LITERAL
                         else (lambda ch, c=c: ch != c)))
        elif op is _c.ANY:
            dotall = bool(flags & re.DOTALL)
            prog.append(("char", lambda ch, dotall=dotall: dotall or ch != "\n"))
        elif op in (_c.LITERAL, _c.NOT_LITERAL, _c.IN):  # IGNORECASE folding, a class
            if op is _c.IN and sum(bounds[1] - bounds[0] + 1 for kind, bounds in av
                                   if kind is _c.RANGE) > MAX_CLASS_SPAN:
                raise _Unsupported("class too wide")
            prog.append(("char", _one_character(op, _key(av), flags)))
        elif op is _c.BRANCH:
            _branch(av[1], flags, prog)
        elif op is _c.SUBPATTERN:
            _group, add, remove, body = av
            _emit(body, _scoped(flags, add, remove), prog)
        elif op is _c.MAX_REPEAT or op is _c.MIN_REPEAT:  # greedy or lazy: the same language
            _repeat(*av, flags, prog)
        elif op is _c.AT:
            prog.append(("at", _one_position(av, flags)))
        else:                     # GROUPREF, ASSERT, ASSERT_NOT, POSSESSIVE_REPEAT, ATOMIC_GROUP...
            raise _Unsupported(str(op))
        if len(prog) > MAX_STATES:
            raise _Unsupported("too large")


def _key(av):
    return tuple(av) if isinstance(av, list) else av


@lru_cache(maxsize=4096)
def _single(op, av, flags: int):
    """`(op, av)` alone, compiled by `re`'s own compiler with `flags`."""
    state = _sre_parse.State()
    state.flags = flags
    item = (op, list(av) if isinstance(av, tuple) and op is _c.IN else av)
    return _sre_compile.compile(_sre_parse.SubPattern(state, [item]), 0)


def _one_character(op, av, flags: int):
    test = _single(op, av, flags).fullmatch
    return lambda ch: test(ch) is not None


def _one_position(where, flags: int):
    test = _single(_c.AT, where, flags).match
    return lambda name, pos: test(name, pos) is not None


def _branch(alternatives, flags: int, prog: list) -> None:
    ends = []
    for i, alt in enumerate(alternatives):
        if i == len(alternatives) - 1:
            _emit(alt, flags, prog)
            break
        split = len(prog)
        prog.append(None)
        _emit(alt, flags, prog)
        ends.append(len(prog))
        prog.append(None)
        prog[split] = ("split", split + 1, len(prog))
    for j in ends:
        prog[j] = ("jmp", len(prog))


def _repeat(low: int, high: int, body, flags: int, prog: list) -> None:
    if low > MAX_STATES or (high != _c.MAXREPEAT and high > MAX_STATES):
        raise _Unsupported("too large")
    for _ in range(low):
        _emit(body, flags, prog)
    if high == _c.MAXREPEAT:                             # body*: split(body, out); body; jmp back
        loop = len(prog)
        prog.append(None)
        _emit(body, flags, prog)
        prog.append(("jmp", loop))
        prog[loop] = ("split", loop + 1, len(prog))
        return
    holes = []
    for _ in range(high - low):                          # body? repeated: each may be skipped
        holes.append(len(prog))
        prog.append(None)
        _emit(body, flags, prog)
        if len(prog) > MAX_STATES:
            raise _Unsupported("too large")
    for h in holes:
        prog[h] = ("split", h + 1, len(prog))


def _closure(prog, starts, name: str, pos: int) -> set[int]:
    """The instructions that consume a character (or accept) reachable from `starts` at `pos`
    without consuming one."""
    out: set[int] = set()
    seen: set[int] = set()
    stack = list(starts)
    while stack:
        pc = stack.pop()
        if pc in seen:
            continue
        seen.add(pc)
        ins = prog[pc]
        kind = ins[0]
        if kind == "split":
            stack.append(ins[2])
            stack.append(ins[1])
        elif kind == "jmp":
            stack.append(ins[1])
        elif kind == "at":
            if ins[1](name, pos):
                stack.append(pc + 1)
        else:
            out.add(pc)
    return out


def _run(prog, name: str) -> bool:
    states = _closure(prog, (0,), name, 0)
    for pos, ch in enumerate(name):
        step = {pc + 1 for pc in states if prog[pc][0] == "char" and prog[pc][1](ch)}
        if not step:
            return False
        states = _closure(prog, step, name, pos + 1)
    return any(prog[pc][0] == "match" for pc in states)


__all__ = ["LOOKUP_STEPS", "MATCH_STEPS", "MAX_STATES", "RUN_STEPS", "bounded", "budget",
           "csearch", "first_match", "fullmatch"]
