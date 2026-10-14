"""Opt-in one-kernel trajectory around frozen original/robust sweep controls.

The frozen overlap coordinator still owns two persistent workers/streams and
performs the same summed raw boundary/metric conversion. Only its serially
prepared quartic solver changes. The controller/retraction/stationary solve and
all physical accuracy controls are delegated unchanged. Existing checkpoints
may be imported only with --resume-from into a distinct fresh output; their
bytes and historical source records remain unchanged.

GPU execution is explicit by running this isolated driver. No existing source
is edited, and importing performs no CUDA work. Larger dimensions remain an
experimental choice until separately qualified on actual GPU physics.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import overlap_sweep_driver as overlap_driver
import persistent_trajectory_overlap as implementation
from warm_sweep_driver import binding,digest

ORIGINAL_PARSE=overlap_driver.BASE_PARSE_ARGS
ORIGINAL_VERIFY=overlap_driver.verify_checkpoint
ORIGINAL_SOURCES=tuple(overlap_driver.OVERLAP_SOURCES)
TRAJECTORY_SOURCES=(*ORIGINAL_SOURCES,'persistent_trajectory_backend.py',
                    'persistent_trajectory_overlap.py','trajectory_sweep_driver.py','generic_layout.py')


def runtime_sources(args):
    controls=getattr(args,'trajectory_controls',{})
    kinetic=controls.get('kinetic','baseline')
    if kinetic not in ('baseline','cooperative'):raise ValueError('invalid recorded trajectory kinetic backend')
    return TRAJECTORY_SOURCES+(('cooperative_arnoldi.py',) if kinetic=='cooperative' else ())


def source_hashes(args):
    hashes=overlap_driver.warm.BASE_SOURCE_HASHES(args)
    hashes.update({name:digest(Path(__file__).with_name(name).read_bytes()) for name in runtime_sources(args)})
    return hashes


def verify_checkpoint(path,requested_args=None):
    before=Path(path).read_bytes();report=json.loads(before);args=argparse.Namespace(**report['controls'])
    expected=(report['runs'][-1] if report.get('runs') else report)['source_sha256']
    core=overlap_driver.warm.BASE_SOURCE_HASHES(args);extras=set(expected)-set(core)
    new_sets=(set(TRAJECTORY_SOURCES),set((*TRAJECTORY_SOURCES,'cooperative_arnoldi.py')))
    if extras in new_sets:
        selected=runtime_sources(args)
        if extras!=set(selected):raise ValueError('trajectory source extras differ from recorded kinetic backend')
        with binding(overlap_driver,'OVERLAP_SOURCES',selected):snapshot=ORIGINAL_VERIFY(path,requested_args)
        controls=getattr(snapshot['args'],'trajectory_controls',{})
        if (controls.get('enabled') is not True or controls.get('whole_trajectory_cooperative_kernel') is not True or
                controls.get('kinetic') not in ('baseline','cooperative') or
                not math.isfinite(controls.get('max_device_seconds',float('nan'))) or controls['max_device_seconds']<=0):
            raise ValueError('trajectory checkpoint lacks complete explicit method/watchdog controls')
        snapshot['trajectory']=True
    else:
        snapshot=ORIGINAL_VERIFY(path,requested_args);snapshot['trajectory']=False
        if getattr(snapshot['args'],'trajectory_controls',{}).get('enabled'):
            raise ValueError('trajectory controls require the complete trajectory source set')
    if digest(before)!=snapshot['checkpoint_sha256'][Path(path).name]:
        raise ValueError('checkpoint changed during trajectory preflight')
    return snapshot


def matching_source_hashes(snapshot,args):
    hashes=overlap_driver.warm.BASE_SOURCE_HASHES(args)
    hashes.update({name:digest(Path(__file__).with_name(name).read_bytes()) for name in snapshot['extras']})
    return hashes


def method_bindings(controls):
    return implementation.candidate_bindings('trajectory',controls['kinetic'],controls['max_device_seconds'])


def _wrapper_options(argv):
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--controller',choices=('original','robust'),default='original')
    parser.add_argument('--resume-from');parser.add_argument('--reset-lbfgs',action='store_true')
    parser.add_argument('--secant-gradient-floor',type=float);parser.add_argument('--max-recoveries',type=int)
    return parser.parse_known_args(argv)


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--kinetic',choices=('baseline','cooperative'))
    parser.add_argument('--max-device-seconds',type=float)
    cli,forwarded=parser.parse_known_args(argv)
    legacy,base_argv=_wrapper_options(forwarded)
    args=ORIGINAL_PARSE(base_argv);overlap_driver.validate_overlap_arguments(args)
    snapshot=verify_checkpoint(legacy.resume_from or args.output,args) if legacy.resume_from or args.resume else None
    if args.resume and (not snapshot or not snapshot['trajectory']):
        raise ValueError('old checkpoint must use explicit --resume-from and a new output for trajectory method')
    prior=getattr(snapshot['args'],'trajectory_controls',{}) if snapshot else {}
    kinetic=cli.kinetic or prior.get('kinetic','baseline')
    maximum=cli.max_device_seconds if cli.max_device_seconds is not None else prior.get('max_device_seconds',30.)
    if not math.isfinite(maximum) or maximum<=0:raise ValueError('max-device-seconds must be positive and finite')
    if args.resume and kinetic!=prior.get('kinetic'):
        raise ValueError('in-place resume cannot change kinetic method; use explicit --resume-from fork')
    controls=dict(enabled=True,whole_trajectory_cooperative_kernel=True,kinetic=kinetic,max_device_seconds=float(maximum))
    def parse(arguments=None):
        out=ORIGINAL_PARSE(arguments);overlap_driver.validate_overlap_arguments(out)
        out.trajectory_controls=dict(controls);return out
    with method_bindings(controls),binding(overlap_driver,'BASE_PARSE_ARGS',parse), \
            binding(overlap_driver,'source_hashes',source_hashes),binding(overlap_driver,'verify_checkpoint',verify_checkpoint):
        return overlap_driver.main(forwarded)


if __name__=='__main__':main()
