"""Isolated dimension-sweep driver with optional stagnation stopping.

Derived from run_experiment.py; the validated D64 driver stays unchanged.
Standalone FP64 accuracy/latency/short-optimization harness.

Example: OPENBLAS_NUM_THREADS=1 python run_experiment.py results/reference_D64
  --backend cupy --mode ode --output results/a100_ode.json

CPU baseline numbers come from existing saved records; this does not run the
Julia optimizer. CPU control, stationary density and boundary solves remain
explicitly identified as part of this hybrid proof of concept.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np

from .numerics import (load_reference, adj, finish_gradient, metric, retract,
                      precondition, stationary_state, SolverFailure)
from .ode_backend import evaluate_phi4, evaluate_quartic, phi_expectation
from .kinetic_backend import kinetic_raw
from .optimizer import OptimizationControls
from .sweep_stopping import optimize_with_progress, ProgressControls


def cpu(x, xp):
    return xp.asnumpy(x) if xp.__name__ == "cupy" else np.asarray(x)


def synchronize(xp):
    if xp.__name__ == "cupy":
        xp.cuda.get_current_stream().synchronize()


def validate_arguments(args):
    ProgressControls(patience=args.stagnation_patience,
        minimum_iterations=args.stagnation_minimum_iterations,
        energy_improvement=args.stagnation_energy_improvement,
        gradient_relative_improvement=args.stagnation_gradient_improvement)
    if not 0 < args.initial_step <= 1:
        raise ValueError("--initial-step must be in (0, 1]")
    if args.quartic_backend == "persistent_trajectory":
        args.device_controller = True
    if args.quartic_backend != "cuda":
        # --packed-z2: the packed Z2 channels; --reduced: the unpacked ones (any state).
        if args.backend != "cupy" or args.mode != "hybrid" or args.packed_z2 == args.reduced:
            raise ValueError(f"--quartic-backend {args.quartic_backend} requires "
                             "--backend cupy --mode hybrid and one of --packed-z2, --reduced")
    if args.graph or args.dense_graph:
        raise ValueError("--graph and --dense-graph: the CUDA-graph integrator was removed")
    if args.device_controller:
        if (args.backend != "cupy" or args.mode != "hybrid" or
                args.quartic_backend != "persistent_trajectory" or args.packed_z2 == args.reduced):
            raise ValueError("--device-controller requires --backend cupy --mode hybrid "
                             "--quartic-backend persistent_trajectory and one of --packed-z2, --reduced")
        if args.fused_rhs:
            raise ValueError("--device-controller is incompatible with --fused-rhs")
        if not 0 < args.window <= 40:
            raise ValueError("--device-controller requires 0 < --window <= 40 for the FP64 source")


def source_hashes(args):
    # The persistent trajectory also uses ode_backend's packed equations and exact
    # pullback; its CUDA source is derived from the persistent_* kernel sources.
    sources = ["sweep_driver.py", "sweep_stopping.py", "ode_backend.py", "kinetic_backend.py",
               "numerics.py", "optimizer.py"]
    if args.quartic_backend == "persistent_trajectory":
        sources += ["persistent_backend.py", "persistent_rows.py", "persistent_rows_cached.py"]
    if args.device_controller:
        sources += ["device_control_backend.py", "jax_source.py"]
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in sources}


def objective(r, k, rho, coupling, args, xp):
    validate_arguments(args)
    if args.device_controller:
        # The persistent trajectory runs only in the overlap objective, which
        # trajectory_sweep_driver binds in place of this function.
        raise ValueError("--quartic-backend persistent_trajectory runs only through trajectory_sweep_driver")
    q = -1j*k-adj(r)@r/2
    synchronize(xp)
    begin = time.perf_counter()
    options = dict(xp=xp, half_window=args.window, n_quad=args.quadrature,
                   atol=args.atol, rtol=args.rtol, max_steps=args.max_steps,
                   progress=lambda status: print(json.dumps({"phase":"ode_progress",**status}),flush=True))
    if args.mode == "ode":
        raw = evaluate_phi4(r, q, rho, coupling, **options)
        stats = {"ode":raw["stats"]}
    else:
        kinetic = kinetic_raw(q, r, rho, args.nodes, xp=xp,
                              tolerance=args.krylov_tolerance, maxdim=args.maxdim,
                              arnoldi_backend=args.arnoldi,
                              fallback_tolerance=args.krylov_ceiling,
                              projection_backend=args.projection)
        print(json.dumps({"phase":"kinetic", "stats":kinetic["stats"]}), flush=True)
        interaction = evaluate_quartic(r, q, rho, coupling, reduced=args.reduced,
                                       packed_z2=args.packed_z2, fused_rhs=args.fused_rhs, **options)
        raw = dict(value=kinetic["value"]+interaction["value"],
                   covector=kinetic["covector"]+interaction["covector"],
                   environment=kinetic["environment"]+interaction["environment"])
        stats = {"kinetic":kinetic["stats"], "quartic":interaction["stats"]}
    synchronize(xp)
    observable_seconds = time.perf_counter()-begin
    covector, environment = cpu(raw["covector"],xp), cpu(raw["environment"],xp)
    if any(x.dtype != np.complex128 for x in (covector,environment)):
        raise SolverFailure("FP64 complex covector/environment required")
    begin = time.perf_counter()
    gradient, boundary_stats = finish_gradient(q,r,rho,covector,environment,
                                               metric_reg=args.metric_reg)
    boundary_seconds = time.perf_counter()-begin
    stats.update(observable_seconds=observable_seconds,
                 cpu_boundary_metric_seconds=boundary_seconds,
                 boundary=boundary_stats,
                 total_seconds=observable_seconds+boundary_seconds)
    return raw["value"],gradient,stats


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("--backend",choices=["cupy","numpy"],default="cupy")
    parser.add_argument("--mode",choices=["ode","hybrid"],default="ode")
    parser.add_argument("--quartic-backend",choices=["cuda","persistent_trajectory"],default="cuda",
                        help="hybrid quartic implementation: cuda (host-driven DOP853) or persistent_trajectory "
                             "(one cooperative CUDA kernel per trajectory; CuPy and packed Z2, through "
                             "trajectory_sweep_driver only; implies the device controller)")
    parser.add_argument("--window",type=float,default=30.)
    parser.add_argument("--quadrature",type=int,default=512)
    parser.add_argument("--atol",type=float,default=1e-13)
    parser.add_argument("--rtol",type=float,default=1e-12)
    parser.add_argument("--nodes",type=int,default=256)
    parser.add_argument("--krylov-tolerance",type=float,default=1e-12)
    parser.add_argument("--maxdim",type=int,default=1024)
    parser.add_argument("--arnoldi",choices=["cpu","gpu"],default="cpu")
    parser.add_argument("--projection",choices=["schur_cpu","hessenberg_gpu"],default="schur_cpu")
    parser.add_argument("--krylov-ceiling",type=float,default=None)
    parser.add_argument("--max-steps",type=int,default=1_000_000)
    parser.add_argument("--metric-reg",type=float,default=1e-3)
    parser.add_argument("--stationary-tolerance",type=float,default=1e-13)
    parser.add_argument("--graph",action="store_true",
                        help="refused: the CUDA-graph integrator was removed (checkpoints record this control)")
    parser.add_argument("--dense-graph",action="store_true",
                        help="refused: removed with --graph (checkpoints record this control)")
    parser.add_argument("--device-controller",action="store_true",
                        help="adaptive controller on the device; implied by --quartic-backend persistent_trajectory")
    parser.add_argument("--reduced",action="store_true")
    parser.add_argument("--packed-z2",action="store_true")
    parser.add_argument("--fused-rhs",action="store_true")
    parser.add_argument("--repeat",type=int,default=1)
    parser.add_argument("--iterations",type=int,default=0)
    parser.add_argument("--perturb",type=float,default=0.)
    parser.add_argument("--initial-step",type=float,default=1.)
    parser.add_argument("--warm-start-line-search",action="store_true")
    parser.add_argument("--gradient-tolerance",type=float,default=3e-5)
    parser.add_argument("--history-size",type=int,default=100)
    parser.add_argument("--line-search-energy-atol",type=float,default=1e-10)
    parser.add_argument("--max-seconds",type=float,default=8*3600)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--stagnation-patience",type=int,default=100)
    parser.add_argument("--stagnation-minimum-iterations",type=int,default=200)
    parser.add_argument("--stagnation-energy-improvement",type=float,default=1e-10)
    parser.add_argument("--stagnation-gradient-improvement",type=float,default=0.05)
    parser.add_argument("--output",default="results/experiment.json")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args=parse_args(argv)
    if args.backend == "cupy":
        import cupy as xp
        device=xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)
        name=device["name"]
        hardware=dict(gpu=name.decode() if isinstance(name,bytes) else name,
                      memory_bytes=device["totalGlobalMem"],cupy=xp.__version__,
                      cuda_runtime=xp.cuda.runtime.runtimeGetVersion(),
                      cuda_driver=xp.cuda.runtime.driverGetVersion())
    else:
        xp=np
        hardware=dict(cpu=platform.processor(),numpy=np.__version__)
    matrices,metadata=load_reference(args.reference)
    if args.backend == "numpy" and metadata["D"] >= 64:
        raise ValueError("D64 CPU reruns are disabled; use the saved reference and GPU backend")
    r,k,rho=matrices["R"],matrices["K"],matrices["rho"]
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes(args)
    reference_hashes={name:hashlib.sha256((Path(args.reference)/(name+".bin")).read_bytes()).hexdigest()
                      for name in ("R","K","rho","saved_gradient")}
    reference_hashes["metadata.toml"]=hashlib.sha256((Path(args.reference)/"metadata.toml").read_bytes()).hexdigest()
    report=dict(hardware=hardware,controls=vars(args),reference=metadata,
                source_sha256=hashes,reference_sha256=reference_hashes,
                measurements=[],history=[],trial_history=[],objective_evaluations=0)
    resume_state=None
    previous_seconds=0.
    if args.resume:
        report=json.loads(output.read_text())
        checkpoint=np.load(output.with_suffix(".npz"))
        if report["reference_sha256"]!=reference_hashes:
            raise ValueError("Resume reference differs from the checkpoint")
        if int(checkpoint["iteration"])!=len(report["history"]):
            raise ValueError("JSON/NPZ checkpoint generations differ")
        r,k,rho,gradient=(checkpoint[name] for name in ("R","K","rho","gradient"))
        energy=float(checkpoint["energy"])
        pairs=list(zip(checkpoint["lbfgs_steps"],checkpoint["lbfgs_changes"],
                       checkpoint["lbfgs_inverse_curvatures"]))
        resume_state=dict(R=r,K=k,rho=rho,gradient=gradient,energy=energy,pairs=pairs,
                          iteration=int(checkpoint["iteration"]),
                          precond_reg=float(checkpoint["precond_reg"]),
                          next_step=float(checkpoint["next_step"]),
                          elapsed_seconds=float(checkpoint["optimizer_seconds"]),
                          evaluations=int(checkpoint["optimizer_evaluations"]))
        previous_seconds=report.get("experiment_seconds",0.)
        report.pop("stop_reason",None)
        report.pop("converged",None)
        # Older checkpoints may predate the per-run provenance list. Preserve
        # their original controls before recording the requested resume choice.
        if not report.get("runs"):
            report["runs"]=[dict(source_sha256=dict(report["source_sha256"]),
                                 controls=dict(report["controls"]), starting_iteration=0)]
        report["controls"]=vars(args)
    report.setdefault("runs",[]).append(dict(source_sha256=hashes,controls=vars(args),
                                            starting_iteration=len(report["history"])))
    experiment_begin=time.perf_counter()

    def save():
        report["experiment_seconds"]=previous_seconds+time.perf_counter()-experiment_begin
        temporary=output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report,indent=2)+"\n")
        temporary.replace(output)

    def save_state(state):
        pairs=state.get("pairs",[])
        empty=np.empty((0,len(r),len(r)),dtype=np.complex128)
        temporary=output.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez(stream,R=state["R"],K=state["K"],rho=state["rho"],
                gradient=state["gradient"],energy=state["energy"],iteration=state.get("iteration",0),
                lbfgs_steps=np.stack([x[0] for x in pairs]) if pairs else empty,
                lbfgs_changes=np.stack([x[1] for x in pairs]) if pairs else empty,
                lbfgs_inverse_curvatures=np.array([x[2] for x in pairs]),
                precond_reg=state.get("precond_reg",1e-4),next_step=state.get("next_step",args.initial_step),
                optimizer_seconds=state.get("elapsed_seconds",0.),
                optimizer_evaluations=state.get("evaluations",0))
        temporary.replace(output.with_suffix(".npz"))

    for i in range(0 if args.resume else args.repeat):
        print(f"Evaluating D={metadata['D']} {args.backend} {args.mode} repetition {i+1}",flush=True)
        energy,gradient,stats=objective(r,k,rho,metadata["coupling"],args,xp)
        report["objective_evaluations"]+=1
        grad_norm=np.sqrt(max(0.,metric(rho,gradient,gradient,args.metric_reg)))
        ref_g=matrices.get("gradient")
        if ref_g is None and metadata.get("saved_gradient_available",True):
            ref_g=matrices.get("saved_gradient")
        ref_e=metadata.get("ode_energy",metadata["saved_energy"]) if metadata.get("saved_energy_available",True) else None
        error=gradient-ref_g if ref_g is not None else None
        measurement=dict(energy=energy,gradient_norm=grad_norm,energy_difference=abs(energy-ref_e) if ref_e is not None else None,
                         gradient_frobenius_difference=float(np.linalg.norm(error)) if error is not None else None,
                         gradient_metric_difference=np.sqrt(max(0.,metric(rho,error,error,args.metric_reg))) if error is not None else None,
                         gradient_norm_difference=abs(grad_norm-metadata["saved_gradient_norm"]) if metadata.get("saved_gradient_norm_available",metadata.get("saved_gradient_available",True)) and "saved_gradient_norm" in metadata else None,
                         stats=stats)
        report["measurements"].append(measurement)
        save_state(dict(R=r,K=k,rho=rho,gradient=gradient,energy=energy))
        save()
        print(json.dumps(measurement),flush=True)

    # A short descent experiment is useful only if the evaluation passed accuracy.
    accuracy=report["measurements"][-1]
    if (accuracy["energy_difference"] is not None and accuracy["energy_difference"]>1e-8) or (accuracy["gradient_metric_difference"] is not None and accuracy["gradient_metric_difference"]>3e-6):
        if args.iterations:
            raise SolverFailure("Accuracy comparison failed; refusing to optimize")
    if args.perturb and not args.resume:
        rng=np.random.default_rng(6427)
        direction=rng.normal(size=r.shape)+1j*rng.normal(size=r.shape)
        d=len(r)//2
        # Preserve the saved Z2 structure when it is present.
        if np.linalg.norm(r[:d,:d])+np.linalg.norm(r[d:,d:])<1e-12*np.linalg.norm(r):
            direction[:d,:d]=0;direction[d:,d:]=0
        direction*=args.perturb*max(1.,np.linalg.norm(r))/np.linalg.norm(direction)
        r,k=retract(r,k,direction,1.)
        q=-1j*k-adj(r)@r/2
        rho,stationary_stats=stationary_state(q,r,tolerance=args.stationary_tolerance)
        energy,gradient,stats=objective(r,k,rho,metadata["coupling"],args,xp)
        report["objective_evaluations"]+=1
        report["perturbed_initial"]=dict(energy=energy,
            gradient_norm=np.sqrt(max(0.,metric(rho,gradient,gradient,args.metric_reg))),
            stats=stats,stationary=stationary_stats)
        save_state(dict(R=r,K=k,rho=rho,gradient=gradient,energy=energy))
        save()
    controls=OptimizationControls(max_iterations=args.iterations,
        gradient_tol=args.gradient_tolerance,max_seconds=args.max_seconds,
        history=args.history_size,metric_reg=args.metric_reg,
        stationary_tolerance=args.stationary_tolerance,
        initial_precond_reg=metadata.get("precond_reg",1e-4),
        initial_step=args.initial_step,warm_start_line_search=args.warm_start_line_search,
        energy_atol=args.line_search_energy_atol)
    evaluation_offset=report["objective_evaluations"]-(resume_state or {}).get("evaluations",0)
    progress_controls=ProgressControls(patience=args.stagnation_patience,
        minimum_iterations=args.stagnation_minimum_iterations,
        energy_improvement=args.stagnation_energy_improvement,
        gradient_relative_improvement=args.stagnation_gradient_improvement)
    for event in optimize_with_progress(r,k,rho,energy,gradient,
                          lambda rr,kk,pp: objective(rr,kk,pp,metadata["coupling"],args,xp),
                          controls,resume_state,progress_controls=progress_controls,
                          initial_gradient_norm=float(np.sqrt(metric(rho,gradient,gradient,args.metric_reg))),
                          prior_history=report["history"]):
        if event["type"]=="trial":
            row={key:value for key,value in event.items() if key not in ("stats","stationary")}
            report["trial_history"].append(row)
            report["objective_evaluations"]=evaluation_offset+event["evaluations"]
            print(json.dumps({"phase":"line_search",**row}),flush=True)
        elif event["type"]=="accepted":
            state=event["state"]
            row={key:value for key,value in event.items() if key not in ("state","type","event")}
            row["phi"]=phi_expectation(state["R"],state["rho"],args.window)
            report["history"].append(row)
            report["objective_evaluations"]=evaluation_offset+state["evaluations"]
            save_state(state)
            save()
            print(json.dumps({"phase":"accepted",**row}),flush=True)
        else:
            state=event["state"]
            report["stop_reason"]=event["reason"]
            report["converged"]=event["converged"]
            report["final_energy"]=state["energy"]
            report["final_gradient_norm"]=state["gradient_norm"]
            report["final_phi"]=phi_expectation(state["R"],state["rho"],args.window)
            report["objective_evaluations"]=evaluation_offset+state["evaluations"]
            save_state(state)
            save()
            print(json.dumps({"phase":"stopped", "reason":event["reason"],
                              "energy":state["energy"],"gradient_norm":state["gradient_norm"],
                              "converged":event["converged"]}),flush=True)
    save()


if __name__=="__main__":
    main()
