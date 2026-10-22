"""Tests for graphflow.ttc: "Scaling Discovery through Test-Time Communication" (Park et al., 2026).

Each section maps to a part of the paper:
  - protocol:  Section 2 + Appendix A.2 (communication prompt, slots, shared records)
  - tasks:     Appendix A.4 (polyomino packing scorer)
  - metrics:   Section 2.1 (best@k), Eq. 1-2 (RHAE), Appendix A.5 (trajectories)
  - theory:    Section 3.4 + Appendix B (Proposition 1)
  - harness:   team@k vs best@k runs, built on graphflow Workflows and Agents
"""

import json
import math
import re
import shutil
import threading
import time
from pathlib import Path

import pytest

from graphflow import ModelResponse, ToolCall
from graphflow.models.mock import MockModel
from graphflow.ttc import (
    COMMUNICATION_PROMPT,
    Budget,
    CallableTask,
    Evaluation,
    PolyominoPackingTask,
    SharedWorkspace,
    SubmissionLedger,
    render_communication_prompt,
    run_independent,
    run_team,
)
from graphflow.ttc.metrics import (
    best_at_k_exact,
    best_so_far,
    independent_best_trajectory,
    matching_pool_size,
    rhae_game,
    rhae_level_score,
)
from graphflow.ttc.tasks.polyomino import InvalidPacking, Piece, packing_quality, transform_cells
from graphflow.ttc.theory import (
    best_success_upper_bound,
    erlang_cdf,
    rate_function,
    simulate_completion,
    team_success_lower_bound,
)

# =========================================================================== #
# Protocol (Section 2, Appendix A.2)
# =========================================================================== #

def test_prompt_template_is_verbatim_from_appendix_a2():
    assert COMMUNICATION_PROMPT.startswith(
        "{n} agents share this container and work the same task in parallel, all with this\n"
        "identical prompt. Search widely without herding, coordinate as you go, and keep\n"
        "improving until time runs out."
    )
    for heading in (
        "1. CLAIM A SLOT AND PICK A DISTINCT APPROACH:",
        "2. ALWAYS BE ACTING.",
        "3. EVALUATE AND RECORD EVERY ATTEMPT.",
        "4. SHARE WITHOUT HERDING.",
        "5. COORDINATE ON SHARED RESOURCES.",
        "6. HIGH BAR TO CONVERGE.",
        "7. NEVER STOP WHILE TIME REMAINS.",
        "8. BREAK PLATEAUS BY CHANGING FAMILY.",
    ):
        assert heading in COMMUNICATION_PROMPT
    assert COMMUNICATION_PROMPT.rstrip().endswith(
        "Do\nnot use the internet, curl, wget, HTTP libraries, or secrets."
    )


def test_rendered_prompt_substitutes_team_size_and_paths(tmp_path):
    ws = SharedWorkspace(tmp_path)
    text = render_communication_prompt(3, ws)
    assert text.startswith("3 agents share this container")
    assert "$(seq 0 2)" in text
    assert f"- slots/approaches: {ws.slots}/" in text
    assert f"- score log: {ws.plateau}" in text
    assert "{" not in text.replace("${", "")  # every template field filled


def test_slot_claims_are_atomic_and_distinct(tmp_path):
    ws = SharedWorkspace(tmp_path)
    claimed: list[int] = []
    lock = threading.Lock()

    def claim():
        slot = ws.claim_slot(8)
        with lock:
            claimed.append(slot)

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(claimed) == list(range(8))
    with pytest.raises(RuntimeError, match="slot"):
        ws.claim_slot(8)
    assert (ws.base / "work-0").is_dir()


def test_approaches_are_visible_to_all_agents(tmp_path):
    ws = SharedWorkspace(tmp_path)
    ws.set_approach(ws.claim_slot(2), "shelf packing; not assuming sorted input")
    ws.set_approach(ws.claim_slot(2), "skyline packing")
    assert ws.approaches() == {0: "shelf packing; not assuming sorted input", 1: "skyline packing"}


def test_score_log_uses_the_paper_format(tmp_path):
    ws = SharedWorkspace(tmp_path)
    ws.log_score(1, 0.883, "skyline")
    line = ws.plateau.read_text().strip()
    assert re.fullmatch(r"\[slot 1 \d\d:\d\d:\d\d\] score=0\.883 family=skyline", line)
    entry = ws.score_log()[0]
    assert (entry.slot, entry.score, entry.family) == (1, "0.883", "skyline")


def test_findings_and_disconfirmations_are_append_only(tmp_path):
    ws = SharedWorkspace(tmp_path)
    ws.append_finding(0, "bottom-left fill gains +0.02 (evidence: run 3)")
    ws.append_finding(1, "WEAK: rotation order may not matter")
    ws.append_disconfirmation(1, "sorting by area does not help (0.81 vs 0.83)")
    findings = ws.read("findings")
    assert findings.index("[slot 0") < findings.index("[slot 1")
    assert "sorting by area" in ws.read("disconfirmations")


def test_plateau_rule_three_non_improving_attempts(tmp_path):
    ws = SharedWorkspace(tmp_path)
    for score in (0.5, 0.6, 0.6, 0.55):
        ws.log_score(0, score, "shelf")
    assert not ws.is_plateaued(0)
    ws.log_score(0, 0.6, "shelf")  # third attempt without strict improvement
    assert ws.is_plateaued(0)
    ws.log_score(0, 0.61, "skyline")
    assert not ws.is_plateaued(0)


def test_plateau_rule_respects_lower_is_better(tmp_path):
    ws = SharedWorkspace(tmp_path)
    for size in (3000, 2900, 2950, 2900, 2901):
        ws.log_score(0, size, "quantize")
    assert ws.is_plateaued(0, higher_is_better=False)


def test_file_lock_serializes_shared_artifacts(tmp_path):
    ws = SharedWorkspace(tmp_path)
    counter = {"value": 0}

    def bump():
        for _ in range(50):
            with ws.lock("submission"):
                value = counter["value"]
                counter["value"] = value + 1

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert counter["value"] == 200


# =========================================================================== #
# Polyomino packing (Appendix A.4)
# =========================================================================== #

def test_transform_reflects_rotates_and_normalizes():
    l_piece = [(0, 0), (0, 1), (0, 2), (1, 2)]
    assert sorted(transform_cells(l_piece, rotation=0, flip=0)) == sorted(l_piece)
    rotated = transform_cells(l_piece, rotation=1, flip=0)
    assert min(x for x, _ in rotated) == 0 and min(y for _, y in rotated) == 0
    assert len(set(rotated)) == 4
    flipped = transform_cells(l_piece, rotation=0, flip=1)
    assert sorted(flipped) == sorted([(1, 0), (1, 1), (1, 2), (0, 2)])


def test_packing_quality_is_cells_over_rectangle_area():
    pieces = [Piece([(0, 0), (1, 0)]), Piece([(0, 0)])]
    output = "3 1\n0 0 0 0\n0 0 2 0\n"
    assert packing_quality(pieces, output) == pytest.approx(3 / 3)
    assert packing_quality(pieces, "4 1\n0 0 0 0\n0 0 3 0\n") == pytest.approx(3 / 4)


@pytest.mark.parametrize(
    "output, reason",
    [
        ("2 1\n0 0 0 0\n0 0 1 0\n", "overlap"),
        ("3 1\n0 0 0 0\n0 0 3 0\n", "outside"),
        ("3 1\n0 0 0 0\n", "piece"),
        ("3 1\n0 5 0 0\n0 0 2 0\n", "rotation"),
        ("garbage", "parse"),
    ],
)
def test_invalid_packings_are_rejected(output, reason):
    pieces = [Piece([(0, 0), (1, 0)]), Piece([(0, 0)])]
    with pytest.raises(InvalidPacking, match=reason):
        packing_quality(pieces, output)


def test_generated_pieces_are_edge_connected_and_deterministic():
    task = PolyominoPackingTask(num_cases=3, min_pieces=5, max_pieces=9, seed=7)
    cases = task.cases()
    assert cases == PolyominoPackingTask(num_cases=3, min_pieces=5, max_pieces=9, seed=7).cases()
    for case in cases:
        assert 5 <= len(case) <= 9
        for piece in case:
            assert 1 <= len(piece.cells) <= 10
            assert piece.is_edge_connected()


NAIVE_PACKER = r"""
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
using namespace std;
int main(){ int n; cin>>n; vector<vector<pair<int,int>>> p(n); int W=0,H=0;
  for(int i=0;i<n;i++){int c;cin>>c;p[i].resize(c);for(auto&q:p[i])cin>>q.first>>q.second;}
  vector<int> xs(n); int x=0;
  for(int i=0;i<n;i++){int w=0,h=0;for(auto&q:p[i]){w=max(w,q.first+1);h=max(h,q.second+1);} xs[i]=x; x+=w; H=max(H,h);}
  W=x; cout<<W<<" "<<H<<"\n"; for(int i=0;i<n;i++) cout<<"0 0 "<<xs[i]<<" 0\n"; }
"""


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++17 compiler")
def test_scorer_compiles_and_scores_a_cpp17_program(tmp_path):
    task = PolyominoPackingTask(num_cases=2, min_pieces=5, max_pieces=8, seed=1)
    source = tmp_path / "solution.cpp"
    source.write_text(NAIVE_PACKER)
    result = task.evaluate(source)
    assert result.valid, result.status
    assert 0 < result.score <= 1
    assert result.details["cases"] == 2


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++17 compiler")
def test_scorer_rejects_programs_that_do_not_compile(tmp_path):
    task = PolyominoPackingTask(num_cases=1, min_pieces=5, max_pieces=5)
    source = tmp_path / "broken.cpp"
    source.write_text("int main( {")
    result = task.evaluate(source)
    assert not result.valid and "compile" in result.status


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++17 compiler")
def test_scorer_is_safe_to_call_from_threads(tmp_path, monkeypatch):
    # preexec_fn is unsafe when the process has threads (TTC agents submit from worker threads).
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    real_run = subprocess.run
    kwargs_seen = []

    def recording_run(*args, **kwargs):
        kwargs_seen.append(kwargs)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)
    task = PolyominoPackingTask(num_cases=1, min_pieces=5, max_pieces=6, seed=2)
    source = tmp_path / "solution.cpp"
    source.write_text(NAIVE_PACKER)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: task.evaluate(source), range(3)))
    assert all(r.valid for r in results), [r.status for r in results]
    assert all(k.get("preexec_fn") is None for k in kwargs_seen)


def test_task_setup_writes_agent_md_without_hidden_cases(tmp_path):
    task = PolyominoPackingTask(num_cases=2, min_pieces=5, max_pieces=6)
    task.setup(tmp_path)
    agent_md = (tmp_path / "AGENT.md").read_text()
    assert "C++17" in agent_md and "submit" in agent_md
    assert not any("case" in p.name for p in tmp_path.rglob("*") if p.is_file() and p.name != "AGENT.md")


# =========================================================================== #
# Metrics (Section 2.1, Eq. 1-2, Appendix A.5)
# =========================================================================== #

def test_best_at_k_exact_without_replacement():
    assert best_at_k_exact(n=64, s=0, k=5) == 0.0
    assert best_at_k_exact(n=10, s=10, k=3) == 1.0
    expected = 1 - math.comb(64 - 3, 5) / math.comb(64, 5)
    assert best_at_k_exact(n=64, s=3, k=5) == pytest.approx(expected)
    with pytest.raises(ValueError):
        best_at_k_exact(n=4, s=1, k=5)


def test_matching_pool_size_finds_smallest_matching_k():
    games = [(64, 3), (64, 0), (64, 10)]
    target = sum(best_at_k_exact(n, s, 13) for n, s in games) / len(games)
    assert matching_pool_size(games, target) == 13


def test_rhae_level_score_eq1():
    assert rhae_level_score(human=10, agent=20) == pytest.approx(0.25)
    assert rhae_level_score(human=10, agent=5) == pytest.approx(1.15)  # capped
    assert rhae_level_score(human=10, agent=None) == 0.0  # never completed


def test_rhae_game_eq2_weights_levels_and_caps_by_completion():
    # 3 levels, weights 1,2,3; only first two completed.
    levels = [(10, 10), (10, 20), (10, None)]
    weighted = (1 * 1.0 + 2 * 0.25 + 3 * 0.0) / 6
    cap = (1 + 2) / 6
    assert rhae_game(levels) == pytest.approx(min(cap, weighted))
    assert rhae_game([(10, 5), (10, 5)]) == pytest.approx(1.0)  # capped at completion fraction 1


def test_best_so_far_trajectory():
    events = [(1.0, 0.5), (2.0, 0.4), (3.0, 0.7), (4.0, None)]
    assert best_so_far(events) == [(1.0, 0.5), (3.0, 0.7)]
    assert best_so_far([(1, 3000), (2, 3100), (3, 2500)], higher_is_better=False) == [(1, 3000), (3, 2500)]


def test_independent_best_trajectory_uses_each_runs_own_clock():
    runs = [
        [(1.0, 0.2), (5.0, 0.9)],
        [(2.0, 0.5)],
    ]
    assert independent_best_trajectory(runs) == [(1.0, 0.2), (2.0, 0.5), (5.0, 0.9)]


# =========================================================================== #
# Theory (Section 3.4, Appendix B)
# =========================================================================== #

def test_rate_function():
    assert rate_function(1.0) == pytest.approx(0.0)
    assert rate_function(2.0) == pytest.approx(1 - math.log(2))
    assert rate_function(0.5) > 0


def test_proposition_1_bounds_hold_in_simulation():
    m, k, lam, alpha = 12, 4, 1.0, 0.5
    sim = simulate_completion(m=m, k=k, lam=lam, alpha=alpha, trials=4000, seed=0)
    assert sim.team_success >= team_success_lower_bound(m, k, alpha) - 0.02
    assert sim.best_success <= best_success_upper_bound(m, k, alpha) + 0.02
    assert sim.team_success > 0.9 > sim.best_success
    assert sim.mean_team_time == pytest.approx(m / (k * lam), rel=0.05)


def test_team_time_never_exceeds_best_time_pointwise():
    sim = simulate_completion(m=5, k=3, lam=1.0, alpha=0.6, trials=500, seed=1)
    assert sim.pointwise_team_le_best


def test_single_stage_gives_no_communication_advantage():
    sim = simulate_completion(m=1, k=3, lam=1.0, alpha=0.6, trials=3000, seed=2)
    assert sim.team_success == pytest.approx(sim.best_success, abs=1e-12)


def test_herding_reduces_team_to_g_groups():
    sim = simulate_completion(m=10, k=6, lam=1.0, alpha=0.4, trials=3000, seed=3, groups=1)
    assert sim.mean_team_time == pytest.approx(10 / 1.0, rel=0.1)


def test_erlang_cdf_matches_known_values():
    assert erlang_cdf(1.0, m=1, rate=1.0) == pytest.approx(1 - math.exp(-1))
    assert erlang_cdf(2.0, m=2, rate=1.0) == pytest.approx(1 - math.exp(-2) * (1 + 2))


def test_proposition_requires_alpha_in_range():
    with pytest.raises(ValueError):
        team_success_lower_bound(m=5, k=2, alpha=0.4)


# =========================================================================== #
# Harness: team@k vs best@k on graphflow
# =========================================================================== #

def _number_task() -> CallableTask:
    """A verifiable toy task: submit a file holding a number; score = the number (capped)."""

    def score(path: Path) -> Evaluation:
        try:
            value = float(path.read_text().strip())
        except ValueError:
            return Evaluation(score=None, valid=False, status="rejected: not a number")
        return Evaluation(score=min(value, 100.0), valid=True)

    return CallableTask(
        name="number",
        instruction="Write a file containing a number and submit it; higher is better.",
        scorer=score,
    )


def _call(name: str, **args) -> ModelResponse:
    call = ToolCall(id=f"{name}-{json.dumps(args, sort_keys=True)}", name=name, args=args)
    return ModelResponse(tool_calls=[call], usage={"completion_tokens": 10})


def _staged_policy(agent: int, k: int):
    """Scripted agent for the paper's staged-improvement model (Section 3.4, Figure 12).

    The task needs successive improvements 0 -> 1 -> 2 -> ...; agent i can only
    find the improvement out of stage s when s % k == i. Alone, an agent stalls
    at the first stage it cannot crack; in a team it continues from the best
    verified result a peer shared in the findings log.
    """

    def decide(current: float) -> ModelResponse:
        if int(current) % k == agent:
            return _call("write_file", path="candidate.txt", content=str(int(current) + 1))
        return ModelResponse(content=f"no improvement found from {current}", usage={"completion_tokens": 10})

    def policy(messages: list) -> ModelResponse:
        team = "SHARED WORKSPACE TOOLS" in messages[0]["content"]
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        last = messages[-1]
        own = [
            float(x)
            for m in tool_msgs if m.get("name") == "submit"
            for x in re.findall(r"score=([0-9.]+)", m["content"])
        ]

        if team and not any(m.get("name") == "claim_slot" for m in tool_msgs):
            return _call("claim_slot", approach=f"stage specialist {agent}")
        if last.get("role") == "user" or last.get("name") == "claim_slot":
            if team:
                return _call("read_shared", which="findings")
            return decide(max(own + [0.0]))
        if last.get("name") == "read_shared":
            shared = [float(x) for x in re.findall(r"verified best=([0-9.]+)", last["content"])]
            return decide(max(shared + own + [0.0]))
        if last.get("name") == "write_file":
            return _call("submit", path="candidate.txt")
        if last.get("name") == "submit" and team:
            score = re.findall(r"score=([0-9.]+)", last["content"])[-1]
            return _call("append_finding", text=f"verified best={score}")
        return ModelResponse(content="turn done", usage={"completion_tokens": 10})

    return policy


def test_team_run_uses_graphflow_workflow_and_shared_workspace(tmp_path):
    result = run_team(
        _number_task(),
        model=lambda i: MockModel(_staged_policy(i, 2)),
        k=2,
        budget=Budget(max_turns=4),
        workdir=tmp_path / "team",
    )
    ws = SharedWorkspace(result.trial_dirs[0])
    assert sorted(ws.approaches()) == [0, 1]
    assert "verified best=" in ws.read("findings")
    assert len(result.agent_reports) == 2
    assert result.best is not None and result.best.valid
    assert all(e.agent in (0, 1) for e in result.ledger_entries)


def _with_latency(policy, seconds: float = 0.005):
    """Simulated model latency, so concurrent agents interleave as real ones do."""

    def slow(messages):
        time.sleep(seconds)
        return policy(messages)

    return slow


def test_verified_progress_sharing_beats_independent_agents(tmp_path):
    # Agents keep working until the wall-clock budget runs out (rule 7). The deadline is
    # shared, so this also checks that the k workers really run concurrently.
    budget = Budget(seconds=1.5)
    team = run_team(_number_task(), model=lambda i: MockModel(_with_latency(_staged_policy(i, 3))), k=3,
                    budget=budget, workdir=tmp_path / "team")
    solo = run_independent(_number_task(), model=lambda i: MockModel(_with_latency(_staged_policy(i, 3))), k=3,
                           budget=budget, workdir=tmp_path / "solo")
    # Alone, only agent 0 can make the first improvement and then it stalls: best@3 = 1.
    # Sharing verified progress lets a different agent supply each next breakthrough.
    assert solo.best.score == 1
    assert team.best.score >= 3


def test_independent_trials_are_isolated(tmp_path):
    result = run_independent(
        _number_task(), model=lambda i: MockModel(_staged_policy(i, 2)), k=2, budget=Budget(max_turns=1),
        workdir=tmp_path / "solo",
    )
    assert len(result.trial_dirs) == 2 and result.trial_dirs[0] != result.trial_dirs[1]
    for d in result.trial_dirs:
        assert not (Path(d) / "team" / "slots").exists()


def test_invalid_submissions_are_recorded_but_never_best(tmp_path):
    ledger = SubmissionLedger(tmp_path / "ledger.jsonl")
    ledger.record(agent=0, slot=0, candidate="a", evaluation=Evaluation(None, False, "rejected"), elapsed=1.0, tokens=10)
    ledger.record(agent=1, slot=1, candidate="b", evaluation=Evaluation(0.7, True), elapsed=2.0, tokens=20)
    ledger.record(agent=1, slot=1, candidate="c", evaluation=Evaluation(0.5, True), elapsed=3.0, tokens=30)
    assert len(ledger.entries()) == 3
    assert ledger.best().candidate == "b"
    reloaded = SubmissionLedger(tmp_path / "ledger.jsonl")
    assert [e.candidate for e in reloaded.entries()] == ["a", "b", "c"]


def test_token_budget_stops_agents(tmp_path):
    result = run_team(
        _number_task(), model=lambda i: MockModel(_staged_policy(i, 1)), k=1,
        budget=Budget(max_turns=50, output_tokens_per_agent=25), workdir=tmp_path / "t",
    )
    report = result.agent_reports[0]
    assert report["stopped"] == "token budget"
    assert report["output_tokens"] <= 25 + 10  # at most one response over the limit
