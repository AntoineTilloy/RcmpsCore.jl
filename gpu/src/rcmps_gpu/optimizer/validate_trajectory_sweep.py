"""Unchanged full accuracy refinement for a stopped trajectory checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import trajectory_sweep_driver as wrapper
import validate_sweep
from warm_sweep_driver import binding,digest


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint');parser.add_argument('initial_reference')
    parser.add_argument('--comparison-reference');parser.add_argument('--output',required=True)
    cli=parser.parse_args(argv);snapshot=wrapper.verify_checkpoint(cli.checkpoint)
    if not snapshot['trajectory']:raise ValueError('validation requires a stopped trajectory checkpoint')
    if Path(cli.initial_reference).resolve()!=Path(snapshot['args'].reference).resolve():
        raise ValueError('validation initial reference differs from checkpoint')
    output=Path(cli.output)
    if output.exists() or output.with_suffix('.npz').exists():raise ValueError('validation output must be new')
    old_argv=sys.argv;caught=None
    try:
        sys.argv=[str(Path(validate_sweep.__file__)),*argv]
        with wrapper.method_bindings(snapshot['args'].trajectory_controls), \
                wrapper.overlap_driver.OverlapObjective() as coordinator, \
                binding(wrapper.overlap_driver.driver,'objective',coordinator), \
                binding(wrapper.overlap_driver.driver,'source_hashes',lambda args:wrapper.matching_source_hashes(snapshot,args)):
            validate_sweep.main()
    except BaseException as error:caught=error
    finally:sys.argv=old_argv
    try:report=json.loads(output.read_bytes()) if output.exists() else dict(evaluations=[],passed=False)
    except Exception as error:
        report=dict(evaluations=[],passed=False,partial_report_read_error=f'{type(error).__name__}: {error}')
    report['trajectory_validation_wrapper_sha256']=digest(Path(__file__).read_bytes())
    report['trajectory_method']=snapshot['args'].trajectory_controls
    if caught is not None:report['validation_exception']=f'{type(caught).__name__}: {caught}'
    try:
        report['checkpoint_unchanged']=(wrapper.verify_checkpoint(cli.checkpoint)['checkpoint_sha256']==snapshot['checkpoint_sha256'])
        report['source_files_unchanged']=(wrapper.matching_source_hashes(snapshot,snapshot['args'])==snapshot['source_sha256'])
    except Exception as error:
        report.update(checkpoint_unchanged=False,source_files_unchanged=False,
                      provenance_guard_error=f'{type(error).__name__}: {error}')
    names=[row.get('name') for row in report.get('evaluations',[])]
    report['complete_five_evaluations']=(names==['standard_cold','standard_warm_1','standard_warm_2','refined_512','refined_1024'])
    report['passed']=bool(caught is None and report.get('passed') and report['complete_five_evaluations'] and
        report['checkpoint_unchanged'] and report['source_files_unchanged'])
    temporary=output.with_suffix('.json.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(output)
    if caught is not None:raise caught
    if not report['passed']:raise RuntimeError('trajectory GPU refinement failed; inspect saved report')


if __name__=='__main__':main()
