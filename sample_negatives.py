#!/usr/bin/env python3
"""Balanced negative sampling (Najm et al.) done correctly, in **accession space**.

Replaces ``generate_negative_samples.py`` for the clean-slate rebuild
(``REBUILD_PLAN.md`` Phase 3).  The old script is left untouched; nothing here
imports it.

What the old one did wrong
--------------------------
``generate_negative_samples.py:190`` resolved targets with
``gene_to_uniprot.get(gene_name)`` over the corrupt ``Mapped_Gene_Name`` map and
dropped the misses with ``if not uniprot_id: continue``.  Its avoid-list
(``:303-319``) was built in *symbol* space, so only 50.11% of its 6,664 entries
pointed at an accession that really carries that gene -- five "negatives" were
in fact known interactions.  And ``:561-566`` contained a fallback that, once no
target still needed negatives, drew from *any* available target: the target-side
quota was silently abandoned.  That branch is deliberately absent here (FIX_PLAN
G8 -- no ``except``/``if not exists`` branch may widen a stage's input set).

The constraint this file implements
-----------------------------------
**Each drug and each target must appear in equal numbers of positive and
negative samples.**  Two consequences the old code never drew:

* a target with **zero** positives must receive **zero** negatives;
* a drug with **zero** positives must receive **zero** negatives.

So the candidate pool is filtered to *structurally eligible* entries -- both
axes present in the corrected positive set -- **before** any drawing happens.
Ineligible entries are surplus by construction; every one is written out with
its reason (``<out>.pool_eligibility.csv``) and the reasons reconcile with the
pool size exactly.  At real scale only 8,063 of the 16,873 pool jobs (47.8%) are
eligible, and *all* of the loss is on the target axis: 523 of 979 docked
accessions have no corrected positives, because the 2025 corruption selected
receptors that nobody targets.

Balance unit
------------
``--balance-unit job`` (default) balances **cavity rows**, which is what the old
code did (one positive row per cavity at ``:201-220``, quotas from
``groupby().size()`` at ``:504-505``, one row per draw at ``:581-595``).  Its
shipped artifact is perfect on rows (1,046/1,046 drugs) and badly skewed on
pairs (338/1,046): 3,982 positive pairs at 3.38 cavities each against 9,856
negative pairs at 1.36.

``--balance-unit pair`` balances distinct ``(drug, accession)`` pairs and then
docks each drawn negative pair at the same cavity depth as the positives of that
target, so the two classes are comparable per pair.  The unit actually used is
recorded in the report and in every ``selected_by`` string, because Phase 3's
STOP gate must state which unit it asserts.

Usage
-----
::

    python3 sample_negatives.py \\
        --index        $NEW/identity/identity_index.parquet \\
        --interactions $NEW/manifests/interactions_resolved.csv \\
        --positives    $NEW/manifests/positives_planned.csv \\
        --pool         $NEW/manifests/negatives_pool.csv \\
        --prefer-pool \\
        --redock-queue $NEW/manifests/redock_queue_negatives.csv \\
        --out          $NEW/manifests/negatives_drawn.csv

Determinism: identical inputs and ``--seed`` produce byte-identical outputs, on
any ``PYTHONHASHSEED``.  Nothing here iterates a ``set``, and no timestamp is
written into any artifact.

Exit codes: 0 ok, 2 usage, 3 input contract violation, 4 invariant violation,
5 ``--require-balance`` and the design could not be balanced.
"""

from __future__ import annotations

import argparse
import heapq
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:  # `python3 sample_negatives.py` from anywhere
    sys.path.insert(0, _REPO_ROOT)

from dock_selectivity.identity import (  # noqa: E402
    IdentityIndex,
    InvariantViolation as _IdentityInvariantViolation,
)
from dock_selectivity.invariants import (  # noqa: E402
    InvariantViolation,
    assert_total_accounting,
    assert_unique_key,
    write_invariant_report,
)

#: ``identity.py`` currently declares its own ``InvariantViolation`` rather than
#: importing the canonical one from ``invariants.py``, so ``IdentityIndex.load``
#: raises a class that ``except InvariantViolation`` does not catch. Catch both.
#: When identity.py adopts the canonical exception this tuple collapses to one
#: class and nothing here changes.
IDENTITY_ERRORS = (InvariantViolation, _IdentityInvariantViolation)
from dock_selectivity.jobid import format_job_dir  # noqa: E402
from dock_selectivity.joins import checked_merge  # noqa: E402
from dock_selectivity.schemas import (  # noqa: E402
    SchemaViolation,
    empty_frame,
    read_checked,
    write_checked,
)

__all__ = [
    "UNIT_JOB",
    "UNIT_PAIR",
    "BALANCE_UNITS",
    "ELIGIBLE",
    "ELIGIBILITY_REASONS",
    "DEFAULT_SEED",
    "SamplingResult",
    "build_avoid_list",
    "compute_quotas",
    "pockets_by_accession",
    "assert_labels_name_their_accession",
    "classify_pool",
    "sample_negatives",
    "write_outputs",
    "build_arg_parser",
    "main",
]

logger = logging.getLogger("sample_negatives")

#: Balance units.  ``job`` = one cavity row; ``pair`` = one (drug, accession).
UNIT_JOB = "job"
UNIT_PAIR = "pair"
BALANCE_UNITS = (UNIT_JOB, UNIT_PAIR)

DEFAULT_SEED = 42

#: Eligibility verdicts for a pool row.  Exactly one applies to every row, and
#: the counts sum to ``len(pool)`` -- checked with ``assert_total_accounting``.
ELIGIBLE = "eligible"
REASON_NOT_IN_INDEX = "accession_not_in_index"
REASON_KNOWN_INTERACTION = "known_interaction"
REASON_BOTH_NO_POSITIVES = "both_axes_have_no_positives"
REASON_TARGET_NO_POSITIVES = "target_has_no_positives"
REASON_DRUG_NO_POSITIVES = "drug_has_no_positives"
REASON_NO_FINAL_RESULTS = "no_final_results"
REASON_DUPLICATE_POCKET = "duplicate_pocket"

#: Precedence, highest first.  A row gets the first reason that applies, so the
#: ``*_no_positives`` counts remain comparable with the reuse analysis.
ELIGIBILITY_REASONS = (
    ELIGIBLE,
    REASON_NOT_IN_INDEX,
    REASON_KNOWN_INTERACTION,
    REASON_BOTH_NO_POSITIVES,
    REASON_TARGET_NO_POSITIVES,
    REASON_DRUG_NO_POSITIVES,
    REASON_NO_FINAL_RESULTS,
    REASON_DUPLICATE_POCKET,
)

SOURCE_POOL = "pool"
SOURCE_NEW = "new"

PHASE_POOL_ONLY = "pool_only"
PHASE_OPEN = "open"
PHASE_REPAIR = "repair"
PHASES = (PHASE_POOL_ONLY, PHASE_OPEN, PHASE_REPAIR)

#: Ceiling on the work the endgame repair may do.  The repair is a bounded
#: improvement, never a correctness requirement: whatever it cannot close is
#: reported as a deficit.
DEFAULT_MAX_REPAIR_SCANS = 200000

_POCKET_COLUMNS = ("fragment_id", "cavity_index")
_PAIR_COLUMNS = ("drugbank_id", "accession")
_JOB_COLUMNS = ("drugbank_id", "accession", "fragment_id", "cavity_index")

_MAX_REPORTED = 20

# Exit codes -- distinct so a runbook can branch on them.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INPUT = 3
EXIT_INVARIANT = 4
EXIT_UNBALANCED = 5


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _pocket_sort_key(pocket: Tuple[str, int]) -> Tuple[int, int, int, str]:
    """Natural order for ``('F10', 3)``: fragment number, then cavity index.

    Lexicographic order would put ``F10`` between ``F1`` and ``F2``, which is
    merely ugly here but is the same class of sloppiness that wrote ``'F1'`` on
    positives and bare ``'1'`` on negatives and made a join match neither.
    """
    fragment, cavity = pocket
    text = str(fragment)
    if len(text) > 1 and text[0] in "Ff" and text[1:].isdigit():
        return (0, int(text[1:]), int(cavity), text)
    return (1, 0, int(cavity), text)


def _as_str(value: Any) -> str:
    return "" if value is None else str(value)


def _pair_set(df: pd.DataFrame) -> FrozenSet[Tuple[str, str]]:
    """``{(drugbank_id, accession)}`` for a frame carrying both columns."""
    return frozenset(
        (str(d), str(a))
        for d, a in zip(df["drugbank_id"].tolist(), df["accession"].tolist())
    )


def _counts_to_dict(series: pd.Series) -> Dict[str, int]:
    """A ``groupby().size()`` result as a plain, insertion-sorted dict."""
    return {str(k): int(v) for k, v in sorted(series.items())}


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def build_avoid_list(interactions: pd.DataFrame) -> FrozenSet[Tuple[str, str]]:
    """The known-interaction avoid-list, **in accession space**.

    Built from every row of ``interactions_resolved.csv`` -- not only the ones
    that became planned positives.  A drug-target pair that is a real
    interaction but has no cavity is still an interaction, and must never be
    drawn as a negative.  The old avoid-list was keyed on
    ``gene_to_uniprot[node_2_name]`` and only 50.11% of its entries pointed at
    an accession that really carries that gene.
    """
    if "drugbank_id" not in interactions.columns or "accession" not in interactions.columns:
        raise SchemaViolation(
            "interactions frame must carry drugbank_id and accession; got "
            "{0}".format(list(interactions.columns)),
            name="build_avoid_list",
        )
    return _pair_set(interactions)


def compute_quotas(
    positives: pd.DataFrame,
    balance_unit: str = UNIT_JOB,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Per-drug and per-target positive counts -- the quotas negatives must match.

    ``job``: one unit per cavity row of ``positives_planned.csv``.
    ``pair``: one unit per distinct ``(drugbank_id, accession)``.

    Returns two dicts keyed in sorted order.  A drug or target absent from the
    positives is absent from its dict, which is exactly how "zero positives ->
    zero negatives" is enforced: there is no key to draw against.
    """
    if balance_unit not in BALANCE_UNITS:
        raise ValueError(
            "balance_unit must be one of {0}, got {1!r}".format(BALANCE_UNITS, balance_unit)
        )
    frame = positives
    if balance_unit == UNIT_PAIR:
        frame = positives[list(_PAIR_COLUMNS)].drop_duplicates()
    drug_quota = _counts_to_dict(frame.groupby("drugbank_id", dropna=False).size())
    target_quota = _counts_to_dict(frame.groupby("accession", dropna=False).size())
    return drug_quota, target_quota


def pockets_by_accession(
    *frames: pd.DataFrame,
) -> Dict[str, Tuple[Tuple[str, int], ...]]:
    """``accession -> ordered (fragment_id, cavity_index)`` over the given frames.

    This is the only pocket universe the sampler knows about, and it is derived
    from evidence: a pocket appears here because a positive was planned for it
    or because a job was physically docked into it.  The sampler never invents a
    cavity index, and it never falls back to "all cavities on disk" -- the
    universe-widening pattern of ``extract_cavities.py:65``.
    """
    grouped: Dict[str, Set[Tuple[str, int]]] = {}
    for frame in frames:
        if frame is None or not len(frame):
            continue
        for accession, fragment, cavity in zip(
            frame["accession"].tolist(),
            frame["fragment_id"].tolist(),
            frame["cavity_index"].tolist(),
        ):
            grouped.setdefault(str(accession), set()).add((str(fragment), int(cavity)))
    return {
        accession: tuple(sorted(pockets, key=_pocket_sort_key))
        for accession, pockets in sorted(grouped.items())
    }


def assert_labels_name_their_accession(
    frame: pd.DataFrame,
    index: IdentityIndex,
    name: str,
    skip_unknown_accessions: bool = False,
) -> Dict[str, Any]:
    """The ``KNOWN_ISSUE.md`` regression assertion, applied to an input manifest.

    Every ``(accession, gene)`` pair must satisfy
    ``gene in index.by_accession(accession).symbols``.  A manifest whose gene
    column names a *different* protein was built against a different (or a
    corrupt) identity index, and nothing downstream of it can be trusted -- 57.7%
    of ``uniprot_gene_mapping.csv`` failed exactly this check.

    The gene column is never *used* by this sampler (labels are re-derived from
    the index), so this is a fail-closed consistency gate on the inputs, not a
    data dependency.
    """
    pairs = frame[["accession", "gene"]].drop_duplicates()
    bad: List[str] = []
    n_unknown = 0
    for accession, gene in zip(pairs["accession"].tolist(), pairs["gene"].tolist()):
        try:
            protein = index.by_accession(accession)
        except KeyError:
            n_unknown += 1
            if skip_unknown_accessions:
                continue
            bad.append("{0} (not in the identity index)".format(accession))
            continue
        if not protein.has_symbol(gene):
            bad.append(
                "{0} labelled {1!r} but its symbols are {2}".format(
                    accession, gene, list(protein.symbols)
                )
            )
    report = {
        "check": "labels_name_their_accession",
        "name": name,
        "n_pairs": int(len(pairs)),
        "n_violations": len(bad),
        "n_unknown_accessions": n_unknown,
        "examples": bad[:_MAX_REPORTED],
        "ok": not bad,
    }
    if bad:
        raise InvariantViolation(
            "{0}: {1} of {2} (accession, gene) pair(s) carry a gene that does not "
            "name that accession: {3}. Rebuild the manifest from the same "
            "identity_index.parquet this run is pinned to.".format(
                name, len(bad), len(pairs), "; ".join(bad[:_MAX_REPORTED])
            ),
            name=name,
            report=report,
        )
    return report


# --------------------------------------------------------------------------
# eligibility
# --------------------------------------------------------------------------


def classify_pool(
    pool: pd.DataFrame,
    drug_quota: Mapping[str, int],
    target_quota: Mapping[str, int],
    avoid: FrozenSet[Tuple[str, str]],
    index: IdentityIndex,
) -> pd.DataFrame:
    """Label every pool row ``eligible`` or give it a reason it is not.

    Returns ``pool`` plus ``eligible`` (bool) and ``eligibility`` (reason)
    columns, one row per input row, in ``old_job_dir`` order.  Nothing is
    dropped here -- the caller writes the whole frame out, so a surplus job is
    accounted for rather than vanishing.

    Precedence (documented because it decides which count a row lands in):

    1. ``accession_not_in_index`` -- the row's receptor is unknown to the pinned
       index, so it cannot be relabelled.
    2. ``known_interaction`` -- contamination.  The 2025 negatives contained 5.
    3. ``*_no_positives`` -- structural ineligibility under balanced sampling.
    4. ``no_final_results`` -- eligible by design, but carries no evidence, so
       salvaging it would reuse nothing; the pocket stays available as a fresh
       docking job.
    5. ``duplicate_pocket`` -- a second old directory for the same
       ``(drug, accession, fragment, cavity)``; keeping both would break the
       primary key of ``negatives_drawn.csv``.
    """
    if not len(pool):
        out = pool.copy()
        out["eligibility"] = pd.Series([], dtype="string")
        out["eligible"] = pd.Series([], dtype=bool)
        return out

    frame = pool.sort_values(
        list(_JOB_COLUMNS) + ["old_job_dir"], kind="mergesort"
    ).reset_index(drop=True)

    drug_quota_frame = pd.DataFrame(
        {
            "drugbank_id": pd.Series(list(drug_quota.keys()), dtype="string"),
            "_drug_quota": pd.Series(list(drug_quota.values()), dtype="Int64"),
        }
    )
    target_quota_frame = pd.DataFrame(
        {
            "accession": pd.Series(list(target_quota.keys()), dtype="string"),
            "_target_quota": pd.Series(list(target_quota.values()), dtype="Int64"),
        }
    )
    # checked_merge, not merge: a duplicated quota key would fan the pool out,
    # which is how parse_prepare.py:1171 gained 410,087 rows without a murmur.
    frame = checked_merge(
        frame,
        drug_quota_frame,
        on="drugbank_id",
        how="left",
        name="pool x drug quota",
    )
    frame = checked_merge(
        frame,
        target_quota_frame,
        on="accession",
        how="left",
        name="pool x target quota",
    )

    known_accessions = set(index.accessions())
    in_index = frame["accession"].astype("object").isin(known_accessions)
    is_known_interaction = pd.Series(
        [
            (str(d), str(a)) in avoid
            for d, a in zip(
                frame["drugbank_id"].tolist(), frame["accession"].tolist()
            )
        ],
        index=frame.index,
        dtype=bool,
    )
    drug_ok = frame["_drug_quota"].notna()
    target_ok = frame["_target_quota"].notna()
    has_results = frame["has_final_results"].fillna(False).astype(bool)
    duplicate = frame.duplicated(subset=list(_JOB_COLUMNS), keep="first")

    reason = pd.Series(ELIGIBLE, index=frame.index, dtype=object)
    # applied lowest-precedence first, so the highest-precedence mask wins
    reason = reason.mask(duplicate, REASON_DUPLICATE_POCKET)
    reason = reason.mask(~has_results, REASON_NO_FINAL_RESULTS)
    reason = reason.mask(drug_ok & ~target_ok, REASON_TARGET_NO_POSITIVES)
    reason = reason.mask(~drug_ok & target_ok, REASON_DRUG_NO_POSITIVES)
    reason = reason.mask(~drug_ok & ~target_ok, REASON_BOTH_NO_POSITIVES)
    reason = reason.mask(is_known_interaction, REASON_KNOWN_INTERACTION)
    reason = reason.mask(~in_index, REASON_NOT_IN_INDEX)

    frame = frame.drop(columns=["_drug_quota", "_target_quota"])
    frame["eligibility"] = reason.astype("string")
    frame["eligible"] = (reason == ELIGIBLE).astype(bool)
    return frame.sort_values("old_job_dir", kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------
# candidate bookkeeping
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Unit:
    """One drawable negative: a pocket, and the old job that already holds it."""

    fragment_id: str
    cavity_index: int
    old_job_dir: Optional[str]

    @property
    def pocket(self) -> Tuple[str, int]:
        return (self.fragment_id, self.cavity_index)


#: Marker returned by the pair-mode book; pairs are expanded to cavity rows after
#: the draw, so the unit itself carries no pocket.
_PAIR_UNIT = _Unit("F0", -1, None)


class _JobBook:
    """Which ``(drug, accession, pocket)`` jobs are still drawable.

    A pool job is always preferred over a fresh one for the same pair -- there
    is never a reason to re-dock a pocket that is already docked.  A pocket is
    handed out at most once per pair, so the primary key of
    ``negatives_drawn.csv`` cannot collide (the old sampler picked cavities with
    ``np.random.choice`` and could emit the same job twice).
    """

    def __init__(
        self,
        eligible_pool: pd.DataFrame,
        pockets: Mapping[str, Sequence[Tuple[str, int]]],
    ) -> None:
        records: List[Tuple[str, str, str, int, str]] = [
            (str(d), str(a), str(f), int(c), str(j))
            for d, a, f, c, j in zip(
                eligible_pool["drugbank_id"].tolist(),
                eligible_pool["accession"].tolist(),
                eligible_pool["fragment_id"].tolist(),
                eligible_pool["cavity_index"].tolist(),
                eligible_pool["old_job_dir"].tolist(),
            )
        ]
        records.sort(key=lambda r: (r[0], r[1], _pocket_sort_key((r[2], r[3])), r[4]))
        self._pool: Dict[Tuple[str, str], List[_Unit]] = {}
        for drug, accession, fragment, cavity, job in records:
            self._pool.setdefault((drug, accession), []).append(
                _Unit(fragment, cavity, job)
            )
        self._pockets = pockets
        self._used: Dict[Tuple[str, str], Set[Tuple[str, int]]] = {}

    def pool_pairs(self) -> FrozenSet[Tuple[str, str]]:
        return frozenset(self._pool)

    def take(
        self,
        drug: str,
        accession: str,
        allow_new: bool,
        consume: bool = True,
    ) -> Optional[_Unit]:
        key = (drug, accession)
        used = self._used.get(key)
        for unit in self._pool.get(key, ()):
            if used is None or unit.pocket not in used:
                if consume:
                    self._used.setdefault(key, set()).add(unit.pocket)
                return unit
        if not allow_new:
            return None
        for pocket in self._pockets.get(accession, ()):
            if used is None or pocket not in used:
                if consume:
                    self._used.setdefault(key, set()).add(pocket)
                return _Unit(pocket[0], pocket[1], None)
        return None

    def available(self, drug: str, accession: str, allow_new: bool) -> bool:
        return self.take(drug, accession, allow_new, consume=False) is not None

    def release(self, drug: str, accession: str, unit: _Unit) -> None:
        """Hand a pocket back, so the endgame repair can re-place it."""
        used = self._used.get((drug, accession))
        if used is not None:
            used.discard(unit.pocket)


class _PairBook:
    """Pair-mode bookkeeping: a ``(drug, accession)`` pair is drawable once."""

    def __init__(
        self,
        eligible_pool: pd.DataFrame,
        pockets: Mapping[str, Sequence[Tuple[str, int]]],
    ) -> None:
        self._pool_pairs = _pair_set(eligible_pool) if len(eligible_pool) else frozenset()
        self._pockets = pockets
        self._drawn: Set[Tuple[str, str]] = set()

    def pool_pairs(self) -> FrozenSet[Tuple[str, str]]:
        return self._pool_pairs

    def take(
        self,
        drug: str,
        accession: str,
        allow_new: bool,
        consume: bool = True,
    ) -> Optional[_Unit]:
        key = (drug, accession)
        if key in self._drawn:
            return None
        if not self._pockets.get(accession):
            return None
        if not allow_new and key not in self._pool_pairs:
            return None
        if consume:
            self._drawn.add(key)
        return _PAIR_UNIT

    def available(self, drug: str, accession: str, allow_new: bool) -> bool:
        return self.take(drug, accession, allow_new, consume=False) is not None

    def release(self, drug: str, accession: str, unit: _Unit) -> None:
        self._drawn.discard((drug, accession))


class _NeedHeap:
    """Targets ordered by remaining quota, largest need first.

    The greedy rule of Najm et al.: always place the next negative on the target
    that is furthest from its quota.  Ties are broken by a per-target value drawn
    once from the seeded RNG (so the order is reproducible but not biased toward
    low accessions), then by accession, so the ordering is total.
    """

    def __init__(self, needs: Dict[str, int], tiebreak: Mapping[str, float]) -> None:
        self._needs = needs
        self._heap: List[Tuple[int, float, str]] = [
            (-need, tiebreak[target], target)
            for target, need in sorted(needs.items())
            if need > 0
        ]
        heapq.heapify(self._heap)

    def take_best(self, accept: Callable[[str], bool]) -> Optional[str]:
        """Pop, decrement and return the neediest acceptable target, or ``None``.

        There is no "if nothing needs negatives, use any target" branch here.
        ``generate_negative_samples.py:561-566`` had one, and it silently
        abandoned the target-side half of the balance constraint.
        """
        deferred: List[Tuple[int, float, str]] = []
        chosen: Optional[str] = None
        while self._heap:
            entry = heapq.heappop(self._heap)
            negated_need, tiebreak, target = entry
            current = self._needs.get(target, 0)
            if current <= 0 or -negated_need != current:
                continue  # stale entry; the live one is elsewhere in the heap
            if accept(target):
                self._needs[target] = current - 1
                if current - 1 > 0:
                    heapq.heappush(self._heap, (-(current - 1), tiebreak, target))
                chosen = target
                break
            deferred.append(entry)
        for entry in deferred:
            heapq.heappush(self._heap, entry)
        return chosen


@dataclass(frozen=True)
class _Draw:
    drugbank_id: str
    accession: str
    unit: _Unit
    phase: str


def _draw_phase(
    book: Any,
    drug_order: Sequence[str],
    need_drug: Dict[str, int],
    need_target: Dict[str, int],
    avoid: FrozenSet[Tuple[str, str]],
    tiebreak: Mapping[str, float],
    allow_new: bool,
    phase: str,
) -> List[_Draw]:
    """One balanced-sampling pass; round-robin over drugs, one draw each.

    Round-robin (rather than draining one drug at a time) is what keeps a
    high-quota drug from consuming every scarce target before the others are
    served.  A drug that cannot draw is dropped from the pass: availability is
    monotonically non-increasing -- target needs only fall and pockets are only
    consumed -- so a stuck drug can never become unstuck later.
    """
    heap = _NeedHeap(need_target, tiebreak)
    draws: List[_Draw] = []
    active = [drug for drug in drug_order if need_drug.get(drug, 0) > 0]
    while active:
        still_active: List[str] = []
        for drug in active:

            def accept(target: str, _drug: str = drug) -> bool:
                return (_drug, target) not in avoid and book.available(
                    _drug, target, allow_new
                )

            target = heap.take_best(accept)
            if target is None:
                continue
            unit = book.take(drug, target, allow_new)
            if unit is None:  # pragma: no cover - accept() just proved otherwise
                raise InvariantViolation(
                    "candidate ({0}, {1}) passed the availability check but "
                    "yielded no unit".format(drug, target),
                    name="_draw_phase",
                )
            draws.append(_Draw(drug, target, unit, phase))
            need_drug[drug] -= 1
            if need_drug[drug] > 0:
                still_active.append(drug)
        active = still_active
    return draws


def _repair(
    book: Any,
    drug_order: Sequence[str],
    need_drug: Dict[str, int],
    need_target: Dict[str, int],
    avoid: FrozenSet[Tuple[str, str]],
    allow_new: bool,
    draws: List[_Draw],
    max_scans: int = DEFAULT_MAX_REPAIR_SCANS,
) -> Tuple[List[_Draw], Dict[str, Any]]:
    """Close endgame deficits the greedy cannot see, by length-3 augmentation.

    The greedy's failure mode is local: drug *D* still needs a negative, target
    *T* still has quota, but ``(D, T)`` is a known interaction, so neither can
    be served even though a feasible assignment exists.  If some donor drug
    *D2* is already placed on a target *T2* that *D* could use, and *D2* could
    instead take *T*, then moving *D2* from *T2* to *T* and giving *T2* to *D*
    adds one unit to both *D* and *T* and changes nothing else.

    Strictly an improvement: every step reduces the total deficit by exactly
    one and preserves every quota, so aborting on the scan budget can only
    leave the greedy's own result.  Whatever remains is a real shortfall and is
    reported in the deficit table -- the pool (or the interaction graph), not
    the algorithm, is then the binding constraint.
    """
    stats: Dict[str, Any] = {
        "n_repaired": 0,
        "n_scans": 0,
        "budget": int(max_scans),
        "budget_exhausted": False,
    }
    live = [True] * len(draws)
    drawn_targets: Dict[str, List[Tuple[int, str]]] = {}
    for position, draw in enumerate(draws):
        drawn_targets.setdefault(draw.drugbank_id, []).append(
            (position, draw.accession)
        )
    donor_order = sorted(drawn_targets)

    def place(drug: str, target: str, phase: str) -> None:
        unit = book.take(drug, target, allow_new)
        if unit is None:  # pragma: no cover - availability was just checked
            raise InvariantViolation(
                "repair could not place ({0}, {1}) after proving it "
                "available".format(drug, target),
                name="_repair",
            )
        draws.append(_Draw(drug, target, unit, phase))
        live.append(True)
        drawn_targets.setdefault(drug, []).append((len(draws) - 1, target))
        need_drug[drug] -= 1
        need_target[target] -= 1

    progressed = True
    while progressed and not stats["budget_exhausted"]:
        progressed = False
        deficit_drugs = [drug for drug in drug_order if need_drug.get(drug, 0) > 0]
        deficit_targets = sorted(t for t, need in need_target.items() if need > 0)
        if not deficit_drugs or not deficit_targets:
            break
        for drug in deficit_drugs:
            if need_drug.get(drug, 0) <= 0:
                continue
            swap: Optional[Tuple[str, str, int, str]] = None
            # Two sweeps: give up a freshly-planned unit before giving up an
            # already-docked one. Balance is the hard constraint, so a pool
            # unit will be sacrificed if that is the only way to close a
            # deficit -- but never merely because it was scanned first.
            for may_release_pool in (False, True):
                if swap is not None or stats["budget_exhausted"]:
                    break
                for donor in donor_order:
                    if donor == drug:
                        continue
                    stats["n_scans"] += 1
                    if stats["n_scans"] > max_scans:
                        stats["budget_exhausted"] = True
                        break
                    absorbing = None
                    for target in deficit_targets:
                        if (
                            need_target.get(target, 0) > 0
                            and (donor, target) not in avoid
                            and book.available(donor, target, allow_new)
                        ):
                            absorbing = target
                            break
                    if absorbing is None:
                        continue
                    for position, released_target in drawn_targets[donor]:
                        if not live[position]:
                            continue
                        if (
                            not may_release_pool
                            and draws[position].unit.old_job_dir is not None
                        ):
                            continue
                        if (drug, released_target) in avoid:
                            continue
                        if book.available(drug, released_target, allow_new):
                            swap = (donor, absorbing, position, released_target)
                            break
                    if swap is not None:
                        break
            if stats["budget_exhausted"] or swap is None:
                if stats["budget_exhausted"]:
                    break
                continue
            donor, absorbing, position, released_target = swap
            released = draws[position]
            book.release(donor, released_target, released.unit)
            live[position] = False
            need_drug[donor] += 1
            need_target[released_target] += 1
            place(donor, absorbing, PHASE_REPAIR)
            place(drug, released_target, PHASE_REPAIR)
            stats["n_repaired"] += 1
            progressed = True
    kept = [draw for position, draw in enumerate(draws) if live[position]]
    return kept, stats


# --------------------------------------------------------------------------
# the sampler
# --------------------------------------------------------------------------


@dataclass
class SamplingResult:
    """Everything one run produced.  ``report`` is JSON-serialisable."""

    drawn: pd.DataFrame
    eligibility: pd.DataFrame
    deficits: pd.DataFrame
    report: Dict[str, Any]
    invariant_reports: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def balance_ok(self) -> bool:
        return bool(self.report["balance"]["ok"])

    @property
    def redock(self) -> pd.DataFrame:
        """The ``source == 'new'`` half -- jobs that must be docked fresh."""
        return self.drawn[self.drawn["source"] == SOURCE_NEW].reset_index(drop=True)


def _expand_pairs(
    draws: Sequence[_Draw],
    positive_pockets: Mapping[str, Sequence[Tuple[str, int]]],
    pool_rows: Mapping[Tuple[str, str, str, int], str],
) -> List[Tuple[str, str, _Unit, str]]:
    """Pair mode: dock each drawn pair at the same cavity depth as the positives.

    The depth comes from ``positives_planned``, not from the pool, so a negative
    pair on target T gets exactly the pockets a positive pair on T gets.  Where a
    pool job already covers one of those pockets it is reused; the rest become
    fresh jobs.
    """
    rows: List[Tuple[str, str, _Unit, str]] = []
    for draw in draws:
        pockets = positive_pockets.get(draw.accession, ())
        if not pockets:  # pragma: no cover - eligibility guarantees positives
            raise InvariantViolation(
                "pair ({0}, {1}) was drawn but its target has no positive "
                "pockets to mirror".format(draw.drugbank_id, draw.accession),
                name="expand_pairs",
            )
        for fragment, cavity in pockets:
            old_job = pool_rows.get(
                (draw.drugbank_id, draw.accession, fragment, cavity)
            )
            rows.append(
                (
                    draw.drugbank_id,
                    draw.accession,
                    _Unit(fragment, cavity, old_job),
                    draw.phase,
                )
            )
    return rows


def sample_negatives(
    index: IdentityIndex,
    interactions: pd.DataFrame,
    positives: pd.DataFrame,
    pool: pd.DataFrame,
    prefer_pool: bool = False,
    seed: int = DEFAULT_SEED,
    balance_unit: str = UNIT_JOB,
    max_repair_scans: int = DEFAULT_MAX_REPAIR_SCANS,
) -> SamplingResult:
    """Draw balanced negatives in accession space.  The core of this module.

    Every frame is expected to satisfy its ``schemas.py`` contract already (use
    :func:`read_checked`).  ``pool`` may be empty, in which case every drawn
    negative is a fresh docking job -- but the pool argument is **required**,
    because "no pool file, so dock everything" is precisely the kind of
    file-presence-driven mode switch that FIX_PLAN G8 bans.
    """
    if balance_unit not in BALANCE_UNITS:
        raise ValueError(
            "balance_unit must be one of {0}, got {1!r}".format(
                BALANCE_UNITS, balance_unit
            )
        )
    if not len(positives):
        raise InvariantViolation(
            "positives_planned is empty. Balanced sampling is defined relative to "
            "a positive set; drawing negatives against nothing would silently "
            "produce an empty, apparently-successful manifest.",
            name="sample_negatives",
        )

    invariants: List[Dict[str, Any]] = []

    # -- identity gates on the inputs -------------------------------------
    known_accessions = set(index.accessions())
    unknown_positive_accessions = sorted(
        {
            str(a)
            for a in positives["accession"].tolist()
            if str(a) not in known_accessions
        }
    )
    if unknown_positive_accessions:
        raise InvariantViolation(
            "{0} accession(s) in positives_planned are not in the identity index "
            "({1}). positives_planned must be built from the same pinned index "
            "this run uses.".format(
                len(unknown_positive_accessions),
                ", ".join(unknown_positive_accessions[:_MAX_REPORTED]),
            ),
            name="positives x index",
        )
    invariants.append(
        assert_labels_name_their_accession(positives, index, "positives_planned labels")
    )
    if len(pool):
        invariants.append(
            assert_labels_name_their_accession(
                pool, index, "negatives_pool labels", skip_unknown_accessions=True
            )
        )

    # -- avoid-list, in accession space -----------------------------------
    avoid = build_avoid_list(interactions)
    positive_pairs = _pair_set(positives)
    orphan_positive_pairs = sorted(positive_pairs - avoid)
    if orphan_positive_pairs:
        raise InvariantViolation(
            "{0} planned positive pair(s) are absent from interactions_resolved, "
            "e.g. {1}. The two manifests disagree, so the avoid-list is not "
            "provably complete over the positives and a negative could be drawn "
            "on a real interaction.".format(
                len(orphan_positive_pairs),
                ", ".join(
                    "{0}/{1}".format(d, a)
                    for d, a in orphan_positive_pairs[:_MAX_REPORTED]
                ),
            ),
            name="positives x interactions",
        )

    # -- quotas ------------------------------------------------------------
    drug_quota, target_quota = compute_quotas(positives, balance_unit)
    total_quota = sum(drug_quota.values())
    if total_quota != sum(target_quota.values()):  # pragma: no cover - arithmetic
        raise InvariantViolation(
            "per-drug and per-target quotas disagree ({0} vs {1}); both count the "
            "same positive units".format(total_quota, sum(target_quota.values())),
            name="quota accounting",
        )

    positive_pockets = pockets_by_accession(positives)
    all_pockets = pockets_by_accession(positives, pool)

    # -- eligibility, BEFORE any drawing ----------------------------------
    eligibility = classify_pool(pool, drug_quota, target_quota, avoid, index)
    eligible_pool = (
        eligibility[eligibility["eligible"]].reset_index(drop=True)
        if len(eligibility)
        else eligibility
    )
    reason_counts = (
        {
            reason: int(count)
            for reason, count in eligibility["eligibility"].value_counts().items()
        }
        if len(eligibility)
        else {}
    )
    invariants.append(
        assert_total_accounting(
            int(len(pool)),
            int(len(eligible_pool)),
            int(len(eligibility)) - int(len(eligible_pool)),
            name="negatives_pool eligibility",
        )
    )

    pool_rows_by_job: Dict[Tuple[str, str, str, int], str] = {}
    if len(eligible_pool):
        for drug, accession, fragment, cavity, job in zip(
            eligible_pool["drugbank_id"].tolist(),
            eligible_pool["accession"].tolist(),
            eligible_pool["fragment_id"].tolist(),
            eligible_pool["cavity_index"].tolist(),
            eligible_pool["old_job_dir"].tolist(),
        ):
            pool_rows_by_job.setdefault(
                (str(drug), str(accession), str(fragment), int(cavity)), str(job)
            )

    # -- draw --------------------------------------------------------------
    rng = random.Random(seed)
    tiebreak = {target: rng.random() for target in sorted(target_quota)}
    drug_order = sorted(drug_quota)
    rng.shuffle(drug_order)

    book: Any
    if balance_unit == UNIT_PAIR:
        book = _PairBook(eligible_pool, positive_pockets)
    else:
        book = _JobBook(eligible_pool, all_pockets)

    need_drug = dict(drug_quota)
    need_target = dict(target_quota)
    draws: List[_Draw] = []
    if prefer_pool:
        draws.extend(
            _draw_phase(
                book,
                drug_order,
                need_drug,
                need_target,
                avoid,
                tiebreak,
                allow_new=False,
                phase=PHASE_POOL_ONLY,
            )
        )
    draws.extend(
        _draw_phase(
            book,
            drug_order,
            need_drug,
            need_target,
            avoid,
            tiebreak,
            allow_new=True,
            phase=PHASE_OPEN,
        )
    )
    draws, repair_stats = _repair(
        book,
        drug_order,
        need_drug,
        need_target,
        avoid,
        allow_new=True,
        draws=draws,
        max_scans=max_repair_scans,
    )

    if balance_unit == UNIT_PAIR:
        rows = _expand_pairs(draws, positive_pockets, pool_rows_by_job)
    else:
        rows = [(d.drugbank_id, d.accession, d.unit, d.phase) for d in draws]

    # -- materialise -------------------------------------------------------
    label_cache: Dict[str, str] = {}

    def label(accession: str) -> str:
        if accession not in label_cache:
            label_cache[accession] = index.label(accession)
        return label_cache[accession]

    records: List[Dict[str, Any]] = []
    phase_counts: Dict[str, int] = {phase: 0 for phase in PHASES}
    for drug, accession, unit, phase in rows:
        source = SOURCE_POOL if unit.old_job_dir is not None else SOURCE_NEW
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        records.append(
            {
                "drugbank_id": drug,
                "accession": accession,
                "gene": label(accession),
                "fragment_id": unit.fragment_id,
                "cavity_index": int(unit.cavity_index),
                "sample_type": "negative",
                "source": source,
                "old_job_dir": unit.old_job_dir,
                "selected_by": (
                    "balanced_negative unit={0} phase={1} seed={2} "
                    "drug_quota={3} target_quota={4} pool_job={5}".format(
                        balance_unit,
                        phase,
                        seed,
                        drug_quota.get(drug, 0),
                        target_quota.get(accession, 0),
                        unit.old_job_dir if unit.old_job_dir is not None else "none",
                    )
                ),
            }
        )

    if records:
        drawn = pd.DataFrame.from_records(records)
        drawn["old_job_dir"] = drawn["old_job_dir"].astype("object").where(
            drawn["old_job_dir"].notna(), None
        )
        drawn = drawn.sort_values(list(_JOB_COLUMNS), kind="mergesort").reset_index(
            drop=True
        )
        drawn = drawn[list(empty_frame("negatives_drawn").columns)]
    else:
        drawn = empty_frame("negatives_drawn")

    invariants.extend(_verify_draw(drawn, avoid, drug_quota, target_quota))

    # -- balance ledger ----------------------------------------------------
    deficits, balance = _balance_ledger(
        drawn, drug_quota, target_quota, label, balance_unit
    )

    n_pool = int((drawn["source"] == SOURCE_POOL).sum()) if len(drawn) else 0
    n_new = int(len(drawn)) - n_pool
    report: Dict[str, Any] = {
        "balance_unit": balance_unit,
        "seed": int(seed),
        "prefer_pool": bool(prefer_pool),
        "phases": (
            [PHASE_POOL_ONLY, PHASE_OPEN] if prefer_pool else [PHASE_OPEN]
        ),
        "identity_index_sha256": index.source_sha256,
        "positives": {
            "n_rows": int(len(positives)),
            "n_pairs": len(positive_pairs),
            "n_drugs": len(drug_quota),
            "n_targets": len(target_quota),
            "n_units": int(total_quota),
        },
        "avoid_list": {
            "n_pairs": len(avoid),
            "n_drugs": len({d for d, _ in avoid}),
            "n_targets": len({a for _, a in avoid}),
        },
        "pool": {
            "n_rows": int(len(pool)),
            "n_eligible": int(len(eligible_pool)),
            "n_ineligible": int(len(eligibility)) - int(len(eligible_pool)),
            "eligible_fraction": (
                round(len(eligible_pool) / float(len(pool)), 6) if len(pool) else 0.0
            ),
            "reasons": {
                reason: int(reason_counts.get(reason, 0))
                for reason in ELIGIBILITY_REASONS
            },
            "n_eligible_pairs": len(_pair_set(eligible_pool))
            if len(eligible_pool)
            else 0,
            "n_eligible_drugs": int(eligible_pool["drugbank_id"].nunique())
            if len(eligible_pool)
            else 0,
            "n_eligible_targets": int(eligible_pool["accession"].nunique())
            if len(eligible_pool)
            else 0,
        },
        "drawn": {
            "n_rows": int(len(drawn)),
            "n_from_pool": n_pool,
            "n_new": n_new,
            "pool_fraction": (
                round(n_pool / float(len(drawn)), 6) if len(drawn) else 0.0
            ),
            "n_pairs": len(_pair_set(drawn)) if len(drawn) else 0,
            "n_drugs": int(drawn["drugbank_id"].nunique()) if len(drawn) else 0,
            "n_targets": int(drawn["accession"].nunique()) if len(drawn) else 0,
            "n_by_phase": dict(phase_counts),
        },
        "repair": repair_stats,
        "balance": balance,
        "contamination": {"n_drawn_known_interactions": 0},
    }

    return SamplingResult(
        drawn=drawn,
        eligibility=eligibility,
        deficits=deficits,
        report=report,
        invariant_reports=invariants,
    )


def _verify_draw(
    drawn: pd.DataFrame,
    avoid: FrozenSet[Tuple[str, str]],
    drug_quota: Mapping[str, int],
    target_quota: Mapping[str, int],
) -> List[Dict[str, Any]]:
    """Assertions that must hold for the output to be usable at all."""
    reports: List[Dict[str, Any]] = []

    contaminated = sorted(_pair_set(drawn) & avoid) if len(drawn) else []
    report = {
        "check": "no_known_interaction_drawn",
        "name": "negatives_drawn contamination",
        "n_drawn_pairs": len(_pair_set(drawn)) if len(drawn) else 0,
        "n_contaminated_pairs": len(contaminated),
        "examples": ["{0}/{1}".format(d, a) for d, a in contaminated[:_MAX_REPORTED]],
        "ok": not contaminated,
    }
    reports.append(report)
    if contaminated:
        raise InvariantViolation(
            "{0} drawn negative pair(s) are known interactions: {1}. The "
            "avoid-list is built in accession space from every row of "
            "interactions_resolved, so this can only mean the drawing code "
            "bypassed it.".format(
                len(contaminated),
                ", ".join("{0}/{1}".format(d, a) for d, a in contaminated[:_MAX_REPORTED]),
            ),
            name="negatives_drawn contamination",
            report=report,
        )

    if len(drawn):
        stray_targets = sorted(
            {
                str(a)
                for a in drawn["accession"].tolist()
                if str(a) not in target_quota
            }
        )
        stray_drugs = sorted(
            {
                str(d)
                for d in drawn["drugbank_id"].tolist()
                if str(d) not in drug_quota
            }
        )
        eligibility_report = {
            "check": "drawn_entities_have_positives",
            "name": "negatives_drawn eligibility",
            "n_targets_without_positives": len(stray_targets),
            "n_drugs_without_positives": len(stray_drugs),
            "examples": (stray_targets + stray_drugs)[:_MAX_REPORTED],
            "ok": not stray_targets and not stray_drugs,
        }
        reports.append(eligibility_report)
        if stray_targets or stray_drugs:
            raise InvariantViolation(
                "balanced sampling drew negatives for {0} target(s) and {1} "
                "drug(s) that have no positives ({2}). Under Najm balancing a "
                "target with zero positives must receive zero "
                "negatives.".format(
                    len(stray_targets),
                    len(stray_drugs),
                    ", ".join((stray_targets + stray_drugs)[:_MAX_REPORTED]),
                ),
                name="negatives_drawn eligibility",
                report=eligibility_report,
            )
        reports.append(
            assert_unique_key(drawn, list(_JOB_COLUMNS), name="negatives_drawn key")
        )
    return reports


def _balance_ledger(
    drawn: pd.DataFrame,
    drug_quota: Mapping[str, int],
    target_quota: Mapping[str, int],
    label: Callable[[str], str],
    balance_unit: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Per-drug and per-target quota / drawn / deficit ledger.

    Every entity with a quota appears, deficit or not, so the file is a complete
    balance statement rather than a list of complaints.

    Counts are in the **declared balance unit**, which the ``unit`` column
    names: cavity rows under ``job``, distinct ``(drug, accession)`` pairs under
    ``pair``.  In pair mode ``from_pool`` therefore counts pairs that reuse at
    least one already-docked job, which is deliberately not the same number as
    ``report['drawn']['n_from_pool']`` (that one counts rows).
    """
    if balance_unit == UNIT_PAIR and len(drawn):
        counted = drawn[list(_PAIR_COLUMNS)].drop_duplicates()
        pool_flag = (
            drawn.assign(_is_pool=drawn["source"] == SOURCE_POOL)
            .groupby(list(_PAIR_COLUMNS))["_is_pool"]
            .any()
            .reset_index()
        )
        counted = counted.merge(pool_flag, on=list(_PAIR_COLUMNS), how="left")
    elif len(drawn):
        counted = drawn.assign(_is_pool=drawn["source"] == SOURCE_POOL)
    else:
        counted = pd.DataFrame(
            {"drugbank_id": [], "accession": [], "_is_pool": []}, dtype=object
        )

    rows: List[Dict[str, Any]] = []
    for axis, column, quota in (
        ("drug", "drugbank_id", drug_quota),
        ("target", "accession", target_quota),
    ):
        if len(counted):
            drawn_counts = counted.groupby(column).size().to_dict()
            pool_counts = (
                counted[counted["_is_pool"].fillna(False).astype(bool)]
                .groupby(column)
                .size()
                .to_dict()
            )
        else:
            drawn_counts, pool_counts = {}, {}
        for key in sorted(quota):
            got = int(drawn_counts.get(key, 0))
            from_pool = int(pool_counts.get(key, 0))
            rows.append(
                {
                    "axis": axis,
                    "key": key,
                    "gene": label(key) if axis == "target" else "",
                    "unit": balance_unit,
                    "quota": int(quota[key]),
                    "drawn": got,
                    "from_pool": from_pool,
                    "from_new": got - from_pool,
                    "deficit": int(quota[key]) - got,
                }
            )
    deficits = pd.DataFrame.from_records(
        rows,
        columns=[
            "axis",
            "key",
            "gene",
            "unit",
            "quota",
            "drawn",
            "from_pool",
            "from_new",
            "deficit",
        ],
    )
    if len(deficits):
        deficits = deficits.sort_values(
            ["axis", "deficit", "key"], ascending=[True, False, True], kind="mergesort"
        ).reset_index(drop=True)

    overshoot = deficits[deficits["deficit"] < 0] if len(deficits) else deficits
    if len(overshoot):  # pragma: no cover - the draw cannot exceed a quota
        raise InvariantViolation(
            "{0} entit(ies) received MORE negatives than positives, e.g. {1}. "
            "Balanced sampling never overshoots a quota.".format(
                len(overshoot), overshoot.head(5).to_dict("records")
            ),
            name="balance overshoot",
        )

    def axis_summary(axis: str) -> Dict[str, Any]:
        part = deficits[deficits["axis"] == axis] if len(deficits) else deficits
        if not len(part):
            return {
                "n_entities": 0,
                "n_balanced": 0,
                "n_short": 0,
                "total_quota": 0,
                "total_drawn": 0,
                "total_from_pool": 0,
                "total_from_new": 0,
                "total_deficit": 0,
                "max_deficit": 0,
                "worst": [],
            }
        short = part[part["deficit"] > 0]
        return {
            "n_entities": int(len(part)),
            "n_balanced": int((part["deficit"] == 0).sum()),
            "n_short": int(len(short)),
            "total_quota": int(part["quota"].sum()),
            "total_drawn": int(part["drawn"].sum()),
            "total_from_pool": int(part["from_pool"].sum()),
            "total_from_new": int(part["from_new"].sum()),
            "total_deficit": int(part["deficit"].sum()),
            "max_deficit": int(part["deficit"].max()),
            "worst": [
                {
                    "key": str(row["key"]),
                    "gene": str(row["gene"]),
                    "quota": int(row["quota"]),
                    "drawn": int(row["drawn"]),
                    "deficit": int(row["deficit"]),
                }
                for _, row in short.head(_MAX_REPORTED).iterrows()
            ],
        }

    drugs = axis_summary("drug")
    targets = axis_summary("target")
    balance = {
        "unit": balance_unit,
        "drugs": drugs,
        "targets": targets,
        "max_abs_imbalance": max(drugs["max_deficit"], targets["max_deficit"]),
        "ok": drugs["total_deficit"] == 0 and targets["total_deficit"] == 0,
    }
    return deficits, balance


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def _sidecar(out_path: str, suffix: str) -> str:
    stem, _ = os.path.splitext(out_path)
    return "{0}.{1}".format(stem, suffix)


def _write_frame(frame: pd.DataFrame, path: str) -> str:
    """Deterministic, atomic-enough CSV for the sidecar artifacts."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def write_outputs(
    result: SamplingResult,
    out_path: str,
    redock_queue_path: Optional[str] = None,
    eligibility_path: Optional[str] = None,
    deficits_path: Optional[str] = None,
    report_path: Optional[str] = None,
) -> Dict[str, str]:
    """Write ``negatives_drawn.csv`` and its ledgers.  Returns the paths written."""
    written: Dict[str, str] = {}
    written["negatives_drawn"] = write_checked(
        result.drawn, out_path, "negatives_drawn"
    )

    eligibility_path = eligibility_path or _sidecar(out_path, "pool_eligibility.csv")
    columns = [
        c
        for c in (
            "old_job_dir",
            "drugbank_id",
            "accession",
            "gene",
            "fragment_id",
            "cavity_index",
            "old_sample_type",
            "old_gene_label",
            "has_final_results",
            "eligible",
            "eligibility",
        )
        if c in result.eligibility.columns
    ]
    written["pool_eligibility"] = _write_frame(result.eligibility[columns], eligibility_path)

    deficits_path = deficits_path or _sidecar(out_path, "deficits.csv")
    written["deficits"] = _write_frame(result.deficits, deficits_path)

    report_path = report_path or _sidecar(out_path, "report.json")
    payload = dict(result.report)
    payload["invariants"] = result.invariant_reports
    written["report"] = write_invariant_report(payload, report_path)

    if redock_queue_path:
        if os.path.exists(redock_queue_path):
            logger.warning(
                "overwriting %s. This file holds the NEGATIVE half of the "
                "re-dock queue only; if it also carried planned positives they "
                "are gone. Write the two halves to separate files and "
                "concatenate them deliberately.",
                redock_queue_path,
            )
        redock = result.redock
        if len(redock):
            queue = pd.DataFrame(
                {
                    "job_id": [
                        format_job_dir(d, a, g, f, c)
                        for d, a, g, f, c in zip(
                            redock["drugbank_id"].tolist(),
                            redock["accession"].tolist(),
                            redock["gene"].tolist(),
                            redock["fragment_id"].tolist(),
                            redock["cavity_index"].tolist(),
                        )
                    ],
                    "drugbank_id": redock["drugbank_id"].tolist(),
                    "accession": redock["accession"].tolist(),
                    "gene": redock["gene"].tolist(),
                    "fragment_id": redock["fragment_id"].tolist(),
                    "cavity_index": redock["cavity_index"].tolist(),
                    "sample_type": ["negative"] * len(redock),
                    "selected_by": redock["selected_by"].tolist(),
                }
            ).sort_values("job_id", kind="mergesort").reset_index(drop=True)
        else:
            queue = empty_frame("redock_queue")
        written["redock_queue"] = write_checked(
            queue, redock_queue_path, "redock_queue"
        )
    return written


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sample_negatives.py",
        description=(
            "Balanced negative sampling (Najm et al.) in accession space. Each "
            "drug and each target receives as many negatives as it has "
            "positives; a target with zero positives receives zero negatives."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes: 0 ok, 2 usage, 3 input contract violation, "
            "4 invariant violation, 5 --require-balance not satisfied."
        ),
    )
    parser.add_argument(
        "--index",
        required=True,
        help="identity/identity_index.parquet (the pinned root artifact)",
    )
    parser.add_argument(
        "--index-sha256",
        default=None,
        help="pin the identity index; a mismatch is a hard error",
    )
    parser.add_argument(
        "--interactions",
        required=True,
        help="manifests/interactions_resolved.csv -- the avoid-list source",
    )
    parser.add_argument(
        "--positives",
        required=True,
        help="manifests/positives_planned.csv -- defines every quota",
    )
    parser.add_argument(
        "--pool",
        default=None,
        help=(
            "manifests/negatives_pool.csv -- already-docked non-interactions. "
            "Either this or --no-pool is REQUIRED: this stage never infers its "
            "mode from which files happen to exist."
        ),
    )
    parser.add_argument(
        "--out", required=True, help="manifests/negatives_drawn.csv to write"
    )
    parser.add_argument(
        "--no-pool",
        action="store_true",
        help=(
            "declare that there is NO pool of already-docked non-interactions, so "
            "every drawn negative is new work. Explicit alternative to --pool; a "
            "genuinely fresh run must say so rather than have it inferred."
        ),
    )
    parser.add_argument(
        "--prefer-pool",
        action="store_true",
        help=(
            "satisfy each drug/target quota from already-docked pool entries "
            "first, then top up with fresh jobs"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed (default {0}); identical inputs + seed -> identical "
        "output".format(DEFAULT_SEED),
    )
    parser.add_argument(
        "--redock-queue",
        default=None,
        help=(
            "write the source='new' half as a redock_queue manifest. NOTE: this "
            "is the NEGATIVE half only; the positive half comes from Phase 4."
        ),
    )
    parser.add_argument(
        "--balance-unit",
        choices=list(BALANCE_UNITS),
        default=UNIT_JOB,
        help=(
            "what must be balanced: 'job' = cavity rows (what the 2025 code "
            "did), 'pair' = distinct (drug, accession), each drawn pair then "
            "docked at the positives' cavity depth. Default: job."
        ),
    )
    parser.add_argument(
        "--require-balance",
        action="store_true",
        help="exit 5 if any drug or target could not be balanced",
    )
    parser.add_argument("--pool-eligibility", default=None, help="override sidecar path")
    parser.add_argument("--deficits", default=None, help="override sidecar path")
    parser.add_argument("--report", default=None, help="override sidecar path")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default INFO)",
    )
    return parser


def _log_summary(result: SamplingResult) -> None:
    report = result.report
    pool = report["pool"]
    drawn = report["drawn"]
    logger.info("balance unit ......... %s (seed %s)", report["balance_unit"], report["seed"])
    logger.info(
        "positives ............ %d units over %d drugs / %d targets (%d pairs)",
        report["positives"]["n_units"],
        report["positives"]["n_drugs"],
        report["positives"]["n_targets"],
        report["positives"]["n_pairs"],
    )
    logger.info(
        "pool ................. %d rows, %d eligible (%.1f%%)",
        pool["n_rows"],
        pool["n_eligible"],
        100.0 * pool["eligible_fraction"],
    )
    for reason in ELIGIBILITY_REASONS:
        count = pool["reasons"].get(reason, 0)
        if count:
            logger.info("    %-28s %d", reason, count)
    logger.info(
        "drawn ................ %d rows: %d from pool, %d to re-dock (%.1f%% reused)",
        drawn["n_rows"],
        drawn["n_from_pool"],
        drawn["n_new"],
        100.0 * drawn["pool_fraction"],
    )
    unit = report["balance"]["unit"]
    for axis in ("drugs", "targets"):
        summary = report["balance"][axis]
        logger.info(
            "%-8s ............ %d/%d balanced, %d short, deficit %d %s "
            "(pool %d / new %d)",
            axis,
            summary["n_balanced"],
            summary["n_entities"],
            summary["n_short"],
            summary["total_deficit"],
            unit + "s",
            summary["total_from_pool"],
            summary["total_from_new"],
        )
    repair = report["repair"]
    if repair["n_repaired"] or repair["budget_exhausted"]:
        logger.info(
            "repair ............... closed %d endgame deficit unit(s) in %d "
            "scan(s)%s",
            repair["n_repaired"],
            repair["n_scans"],
            " (scan budget exhausted)" if repair["budget_exhausted"] else "",
        )
    if not result.balance_ok:
        logger.warning(
            "design NOT fully balanced: %d unit(s) short. Worst offenders:",
            report["balance"]["drugs"]["total_deficit"],
        )
        for axis in ("drugs", "targets"):
            for entry in report["balance"][axis]["worst"]:
                logger.warning(
                    "    %-7s %-10s %-10s quota %5d drawn %5d deficit %5d",
                    axis[:-1],
                    entry["key"],
                    entry["gene"],
                    entry["quota"],
                    entry["drawn"],
                    entry["deficit"],
                )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        index = IdentityIndex.load(args.index, expect_sha256=args.index_sha256)
        interactions = read_checked(args.interactions, "interactions_resolved")
        positives = read_checked(args.positives, "positives_planned")
        if bool(args.pool) == bool(args.no_pool):
            logger.error(
                "pass exactly one of --pool <negatives_pool.csv> or --no-pool. "
                "This stage will not guess whether prior results exist.")
            return 2
        if args.no_pool:
            pool = empty_frame("negatives_pool")
            logger.info("--no-pool: every drawn negative will be new work")
        else:
            pool = read_checked(args.pool, "negatives_pool")
    except (FileNotFoundError, SchemaViolation) as exc:
        logger.error("input contract violation: %s", exc)
        return EXIT_INPUT
    except IDENTITY_ERRORS as exc:
        logger.error("identity index rejected: %s", exc)
        return EXIT_INPUT

    try:
        result = sample_negatives(
            index=index,
            interactions=interactions,
            positives=positives,
            pool=pool,
            prefer_pool=args.prefer_pool,
            seed=args.seed,
            balance_unit=args.balance_unit,
        )
    except IDENTITY_ERRORS as exc:
        logger.error("invariant violation: %s", exc)
        return EXIT_INVARIANT

    try:
        written = write_outputs(
            result,
            out_path=args.out,
            redock_queue_path=args.redock_queue,
            eligibility_path=args.pool_eligibility,
            deficits_path=args.deficits,
            report_path=args.report,
        )
    except (SchemaViolation,) + IDENTITY_ERRORS as exc:
        logger.error("refusing to write a manifest that fails its contract: %s", exc)
        return EXIT_INVARIANT

    _log_summary(result)
    for name in sorted(written):
        logger.info("wrote %-18s %s", name, written[name])

    if args.require_balance and not result.balance_ok:
        logger.error(
            "--require-balance: %d drug-unit(s) and %d target-unit(s) short; see %s",
            result.report["balance"]["drugs"]["total_deficit"],
            result.report["balance"]["targets"]["total_deficit"],
            written["deficits"],
        )
        return EXIT_UNBALANCED
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
