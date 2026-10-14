import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnfplace.baselines import HEURISTICS, GreedyEnergy  # noqa: E402
from vnfplace.env import EnvConfig, FogEnv, run_policy  # noqa: E402
from vnfplace.policy import ENCODERS, PlacementPolicy  # noqa: E402
from vnfplace.topology import build_infrastructure, list_real_topologies  # noqa: E402
from vnfplace.traffic import generate_trace  # noqa: E402


@pytest.fixture(scope="module")
def infra():
    return build_infrastructure("fog", seed=1)


def test_real_topologies_load():
    for name in list_real_topologies():
        I = build_infrastructure(f"real:{name}", seed=0)
        assert I.n >= 10 and (I.sp_delay >= 0).all()


def test_trace_reproducible(infra):
    a = generate_trace(infra, 50, 1.0, seed=3)
    b = generate_trace(infra, 50, 1.0, seed=3)
    assert [r.to_dict() for r in a] == [r.to_dict() for r in b]


@pytest.mark.parametrize("name", list(HEURISTICS))
def test_capacity_and_sla_never_violated(infra, name):
    """Full mask: no capacity overrun at any time and every accepted request meets its SLA."""
    env = FogEnv(infra, EnvConfig(mask_mode="full"))
    trace = generate_trace(infra, 400, 1.3, seed=11)
    pol = HEURISTICS[name](0)

    def checked(env, cand):
        assert np.all(env.used_cpu <= infra.cap_cpu * env.cfg.u_max + 1e-6)
        assert np.all(env.used_ram <= infra.cap_ram + 1e-6)
        assert np.all(env.used_bw <= infra.cap_bw + 1e-6)
        return pol(env, cand)

    rows, _ = run_policy(env, trace, checked)
    for r in rows:
        if r["accepted"]:
            assert r["latency_ms"] <= r["max_latency_ms"] + 1e-6
        else:
            assert r["reject_reason"] in ("no_candidate", "dead_end")


def test_resources_released(infra):
    env = FogEnv(infra, EnvConfig())
    trace = generate_trace(infra, 300, 1.0, seed=5)
    run_policy(env, trace, GreedyEnergy(0))
    env.advance(trace[-1].t_arrival + 1e7)
    assert not env.services
    assert np.allclose(env.used_cpu, 0, atol=1e-6) and np.allclose(env.used_bw, 0, atol=1e-6)
    assert (env.n_hosted == 0).all()


def test_energy_integral_idle(infra):
    env = FogEnv(infra, EnvConfig(sleep_frac=0.1))
    env.advance(100.0)
    expected = 0.1 * infra.p_idle[~infra.always_on].sum() * 100.0 + infra.p_idle[infra.always_on].sum() * 100.0
    assert env.energy_j == pytest.approx(expected)


def test_resource_mask_reproduces_r6_failures(infra):
    """With the original CPU/RAM-only mask, bandwidth/latency failures appear only after placement."""
    env = FogEnv(infra, EnvConfig(mask_mode="resource"))
    trace = generate_trace(infra, 400, 1.3, seed=11)
    rng = np.random.default_rng(0)
    _, s = run_policy(env, trace, lambda e, c: int(rng.choice(np.flatnonzero(c.mask))))
    assert s["reject_reasons"].get("latency", 0) + s["reject_reasons"].get("bandwidth", 0) > 0


def test_ilp_not_worse_than_greedy_energy(infra):
    pytest.importorskip("ortools")
    from vnfplace.ilp import PerRequestILP
    trace = generate_trace(infra, 120, 1.0, seed=2)
    env = FogEnv(infra, EnvConfig())
    ilp = PerRequestILP(time_limit_s=5)
    for req in trace:
        env.begin(req)
        cand = env.candidates()
        if not cand.mask.any():
            env.abort("no_candidate")
            continue
        snap = env._snapshot()
        # Greedy-energy marginal power on the same state.
        g, g_ok = 0.0, True
        while env.j < req.length:
            c = env.candidates()
            if not c.mask.any():
                g_ok = False
                break
            n = int(np.argmin(np.where(c.mask, c.inc_power, np.inf)))
            g += c.inc_power[n]
            env._reserve_step(n)
        env._restore(snap)
        plan = ilp._solve(env)
        if g_ok:
            assert plan is not None
        if plan is not None:
            p = 0.0
            for n in plan:
                c = env.candidates()
                assert c.mask[n], "ILP plan must pass the sound mask"
                p += c.inc_power[n]
                env._reserve_step(n)
            env._restore(snap)
            if g_ok:
                # Per-request objective: ILP is optimal (up to integer rounding).
                assert p <= g + 0.05
        env.abort("test")


@pytest.mark.parametrize("enc", list(ENCODERS))
@pytest.mark.parametrize("actor", ["shared", "separate"])
def test_policy_shapes_and_masking(infra, enc, actor):
    env = FogEnv(infra, EnvConfig())
    req = generate_trace(infra, 1, 1.0, seed=0)[0]
    env.begin(req)
    cand = env.candidates()
    x, ef, ctx = env.observe(cand)
    B = 3
    pol = PlacementPolicy(x.shape[1], ef.shape[1], ctx.shape[0], encoder=enc, actor=actor)
    X = torch.from_numpy(np.repeat(x[None], B, 0))
    EF = torch.from_numpy(np.repeat(ef[None], B, 0))
    C = torch.from_numpy(np.repeat(ctx[None], B, 0))
    M = torch.from_numpy(np.repeat(cand.mask[None], B, 0))
    logits, v = pol(X, EF, C, M, torch.tensor([0, 1, 2]), torch.from_numpy(env.edge_index.copy()))
    assert logits.shape == (B, infra.n) and v.shape == (B,)
    probs = torch.softmax(logits, -1)
    assert torch.all(probs[~M] < 1e-6)
