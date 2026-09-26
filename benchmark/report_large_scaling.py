"""Render the large-D benchmark CSV: python report_large_scaling.py RESULTS.csv."""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

source = Path(sys.argv[1]).resolve()
with source.open() as stream:
    rows = list(csv.DictReader(stream))
out = source.parent
good = {(r['case'], r['stage'], r['backend']): r for r in rows if r['status'] == 'ok'}
labels = {'random32': 'Random D32', 'random48': 'Random D48', 'random64': 'Random D64',
          'saved_phi4_28': 'Saved phi4 g2 D28', 'saved_shg_32': 'Saved shg D32 stress state'}
cases = list(dict.fromkeys(r['case'] for r in rows))
lines = ['# Large bond dimensions: matrix-free kinetic evaluation', '',
         'These are complete value-and-gradient evaluations at fixed states, not full',
         'optimizations. The saved shg state is evaluated with the phi4 g=2 objective;',
         'it is a stress test and is not a g=2 optimum.', '',
         '| State | Threaded ODE (s) | Threaded Krylov + interaction ODE (s) | Speedup |',
         '|---|---:|---:|---:|']
for case in cases:
    a = good.get((case, 'full_phi4_g2', 'ode'))
    b = good.get((case, 'full_phi4_g2', 'krylov'))
    if a and b:
        ta, tb = float(a['median_seconds']), float(b['median_seconds'])
        lines.append(f'| {labels.get(case, case)} | {ta:.3f} | {tb:.3f} | {ta/tb:.2f}x |')
lines += ['', '| State | Largest basis / D² | All four bases (MiB) | Angular nodes | Largest true residual |',
          '|---|---:|---:|---:|---:|']
for case in cases:
    r = good.get((case, 'kinetic_adaptive_diagnostic', 'krylov'))
    if r:
        k = max(int(part.split(':')[1]) for part in r['basis_dimensions'].split(';'))
        lines.append(f"| {labels.get(case, case)} | {k} / {int(r['D'])**2} | "
                     f"{float(r['basis_bytes'])/2**20:.2f} | {r['accepted_nodes']} | "
                     f"{float(r['true_relative_residual']):.2e} |")
lines += ['', 'Basis storage excludes the state, temporary angular solutions, and ODE trajectories;',
          'it is not peak process memory. CSV allocation counts are cumulative allocations per',
          'evaluation, not retained memory. The physical transfer matrix is never formed by',
          'the Krylov backend.', '',
          '| State | Independent kinetic reference | Absolute energy difference | Full gradient difference (Frobenius) |',
          '|---|---|---:|---:|']
for case in cases:
    for stage, backend, label in [('kinetic_dense_reference', 'resolvent', 'Adaptive dense resolvents'),
                                  ('kinetic_tight_ODE_reference', 'ode', 'Tight ODE, L30')]:
        r = good.get((case, stage, backend))
        if r:
            lines.append(f"| {labels.get(case, case)} | {label} | {float(r['energy_difference']):.2e} | "
                         f"{float(r['gradient_difference']):.2e} |")
failures = [r for r in rows if r['status'] != 'ok']
lines += ['', f'Failed or incomplete stages: {len(failures)}.', '']
for r in failures:
    lines.append(f"- {r['case']}, {r['stage']}, {r['backend']}: {r['message']}")
lines += ['', '## Protocol and interpretation', '',
          'Intel i7-9750H, Julia 1.12.7, six compute threads, one BLAS thread and one GC',
          'thread. Each timing includes an untimed warmup and reports the median of three',
          'samples. Processes ran sequentially without other research benchmark jobs.',
          'Both full evaluators use `parallel=:both`. Solver contexts are rebuilt every',
          'evaluation, as required when an optimizer changes the state.', '',
          'The half-window remains 20. Random matrices use seed 6100+D, atol=1e-11,',
          'rtol=1e-10 and 128 nodes per half. Saved states use atol=1e-13, rtol=1e-12',
          'and 192 nodes. The interaction follows the same ODE in both methods.',
          'Krylov kinetics integrate the source analytically on the infinite line,',
          'with true relative residual tolerance 1e-12 and a maximum basis of 512.',
          'Angular refinement starts at 32 nodes and doubles to at most 512, checking',
          'energy and the complete stationary-response gradient (atol=1e-11, rtol=1e-10).', '',
          'Dense reference factorizations are restricted to D<=32. D48 and D64 use an',
          'independently tightened kinetic ODE reference at L30, atol=1e-13, rtol=1e-12,',
          '256 nodes. These single validation evaluations are not timing comparisons.',
          'Node differences are empirical convergence checks; shifted residuals do not',
          'by themselves bound full gradient error in a nonnormal, small-gap problem.', '',
          'The complete ODE/Krylov evaluator energies differ by at most 2.21e-12.',
          'Their full gradients differ by at most 2.93e-8, on the saved D28 state;',
          'the independent dense kinetic comparison there differs by only 3.64e-10.',
          'Thus reference discretization matters when interpreting a near-stationary',
          'gradient. Neither comparison bounds the shared interaction ODE error.', '',
          'For a basis of size k, construction costs O(kD³+k²D²), the projected Schur',
          'factorization O(k³), and persistent storage O(kD²+k²) per right-hand side.',
          'Angular reconstruction and physical contractions also contribute. Basis size',
          'depends on the state: these measurements do not prove fixed-k asymptotic scaling.', '',
          '![Large-D evaluation and basis storage](large_scaling.png)', '',
          f'[Validated measurements]({source.name}) · [Harness](../large_scaling.jl) · '
          'Derivation: `research/structured_observables/README.md` in the companion',
          'research repository rcmps-phi4-z2', '',
          'In the September 2026 record of this benchmark (`benchmark/large_scaling/` in',
          'the companion research repository rcmps-phi4-z2), a benchmark-only Julia',
          'closure capture initially misreported the diagnostic kinetic energy. That',
          'record keeps the original timing CSV and independently rerun corrected',
          'diagnostics; its `benchmark/validate_large_scaling.py` verifies agreement',
          'with the original timed kinetic values and emits the validated CSV.',
          'Timings, full-objective values, gradient comparisons, bases and residuals',
          'were unaffected. The current harness includes the reporting fix.', '',
          'The default run covers the first four cases. Set `RCMPS_LARGE_CASES=saved_shg_32`',
          'to opt into the expensive stress case; use an external process time limit',
          'if needed. Its default Krylov residual failure is retained above.', '',
          '```sh',
          "JULIA_LOAD_PATH='@:@stdlib' julia --project=benchmark --threads=6 --gcthreads=1 benchmark/large_scaling.jl",
          '```', '']
(out / 'README.md').write_text('\n'.join(lines))

random = [c for c in cases if c.startswith('random') and (c, 'full_phi4_g2', 'krylov') in good]
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), layout='constrained')
ds = [int(good[c, 'full_phi4_g2', 'krylov']['D']) for c in random]
for backend, label in [('ode', 'ODE'), ('krylov', 'Krylov + interaction ODE')]:
    axes[0].plot(ds, [float(good[c, 'full_phi4_g2', backend]['median_seconds']) for c in random], 'o-', label=label)
axes[0].set(xlabel='Bond dimension', ylabel='Complete value + gradient (s)', xticks=ds)
axes[0].legend(frameon=False, fontsize=8)
axes[1].plot(ds, [float(good[c, 'kinetic_adaptive_diagnostic', 'krylov']['basis_bytes'])/2**20 for c in random], 'o-', color='#427745')
axes[1].set(xlabel='Bond dimension', ylabel='All four Arnoldi bases (MiB)', xticks=ds)
for ax in axes:
    ax.grid(alpha=.2)
fig.suptitle('Random states: fixed accuracy, six compute threads, BLAS = 1', fontsize=11)
fig.savefig(out / 'large_scaling.png', dpi=180)
fig.savefig(out / 'large_scaling.pdf')
