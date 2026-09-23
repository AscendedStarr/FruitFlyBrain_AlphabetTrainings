"""Where does the training time actually go, and would a GPU help?

The circuit is tiny - 35 receptors, 512 Kenyon cells, 27 MBONs, and a single
27x512 weight matrix (13,824 plastic synapses). That is roughly four orders of
magnitude below the size where GPU kernels pay for themselves, so the question
is not "is CUDA faster" but "what is the CPU actually spending its time on".

This measures the two candidate bottlenecks directly:

* the per-timestep Python loop in ``MushroomBody.present``
* the linear algebra inside it

If the loop dominates, the fix is batching on the CPU and a GPU cannot help.
"""

from __future__ import annotations

import time

import torch

# Repo root on sys.path, so this script still finds the package from scripts/.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from flybrain import Config
from flybrain.trainer import FlyBrain

cfg = Config()
brain = FlyBrain(cfg)
spikes = brain._encode("A", False)

T = cfg.t_stim


def bench(fn, n: int) -> float:
    fn()  # warm up
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


print(f"device            : cpu (torch {torch.__version__})")
print(f"threads           : {torch.get_num_threads()}")
print(f"plastic synapses  : {brain.mb.mbon.w.numel():,}")
print(f"timesteps/trial   : {T}\n")

t_full = bench(lambda: brain.mb.present(spikes), 200)
print(f"full present()      {t_full * 1e3:8.3f} ms/trial")

t_pn = bench(lambda: brain.mb.al.rate_vector(spikes), 200)
print(f"  antennal lobe     {t_pn * 1e3:8.3f} ms")

t_kc = bench(lambda: brain.mb.kc.encode(brain.mb.pn_rate), 200)
print(f"  kenyon encode     {t_kc * 1e3:8.3f} ms")

t_enc = bench(lambda: brain._encode("A", False), 200)
print(f"  spike encoding    {t_enc * 1e3:8.3f} ms")


def _one_step() -> None:
    kc_spk = brain.mb.kc.spike_step(brain.mb.gen)
    brain.mb.mbon.step(kc_spk)


t_step = bench(_one_step, 200)
print(f"  one timestep      {t_step * 1e3:8.3f} ms  x{T} = {t_step * T * 1e3:.2f} ms")
print(f"  -> loop share       {t_step * T / t_full:8.1%} of the trial")

# The matmul on its own, to show how little of the step is real arithmetic.
x = torch.rand(cfg.n_kc)
w = brain.mb.mbon.w
t_mm = bench(lambda: w @ x, 2000)
print(f"\n  27x512 matmul     {t_mm * 1e6:8.2f} us")
print(f"  => arithmetic is    {t_mm / t_step:8.1%} of a timestep "
      f"(the rest is dispatch overhead)")

# Batched: all T timesteps of Poisson sampling in one call.
p = (brain.mb.kc.activation * cfg.kc_rate_gain).clamp(max=cfg.kc_max_rate)
t_batch = bench(lambda: (torch.rand(T, cfg.n_kc, generator=brain.mb.gen) < p), 500)
print(f"\nbatched Poisson ({T} steps) {t_batch * 1e3:8.3f} ms "
      f"vs {t_step * T * 1e3:.2f} ms looped  "
      f"=> {t_step * T / t_batch:.1f}x from batching alone")

print("\nepoch cost at 108 trials:")
print(f"  current   {t_full * 108:8.2f} s")
print(f"  if loop were free {t_full * 108 * (1 - t_step * T / t_full):8.2f} s")
