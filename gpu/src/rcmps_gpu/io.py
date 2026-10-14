"""State files shared with the Julia package: reading and writing them, for both subpackages.

Both formats are read into one normalized StateRecord. NumPy and tomllib only (h5py is
imported by read_jld2); importing this module never imports CuPy.

JLD2 schema 1 of RcmpsCore (rcmps_store and rcmps_load, src/storage.jl), read by
read_jld2 with h5py:
    schema_version = 1; R, K (Matrix{ComplexF64}); model (String); coupling (ComplexF64,
    real here); mass (Float64, optional: 1.0 when absent, as rcmps_load reads files
    written before the model had a mass); e0; grad; info (String); integration
    (n_quad, max_point, atol, rtol, maxiters); manifold (metric_reg,
    precond_reg_prefactor, precond_reg). JLD2 stores Julia's column-major matrices, so
    the HDF5 view is the transpose. rho is not stored: rcmps_load recomputes it.

Portable state directory, "rcmps-portable-state" format 1, written by write_portable and read
by read_portable:

    metadata.toml (UTF-8 TOML)
      format = "rcmps-portable-state"   required
      format_version = 1                required, integer
      D = 16                            required, positive integer: the bond dimension
      model = "phi4"                    required, the model name of RcmpsCore
      coupling = 2.0                    required, real
      mass = 1.0                        optional, real, default 1.0; model "phi4" has no
                                        mass parameter (its normal-ordering mass is 1)
                                        and requires 1.0
      energy = -0.156                   optional, real: energy density at [integration]
      gradient_norm = 6.4e-6            optional, real: sqrt(metric(g, g)) of the gradient,
                                        with metric_reg of [manifold]
      validation_passed = true          optional: set by a producer that validated the state
      producer = "..."                  optional: program and version that wrote the directory
      source = "..."                    optional: where the state comes from (a file name or
                                        a description, not a path of the producing machine)
      info = "..."                      optional: the info string of RcmpsCore's State
      [integration]                     optional: IntegratorParameters of the state, those at
        n_quad (integer), max_point,      which energy and gradient were computed (for a state
        atol, rtol (reals),               without them, those it is meant to be used with)
        maxiters (integer, optional)
      [manifold]                        optional: ManifoldParameters of the state
        metric_reg, precond_reg (reals),
        precond_reg_prefactor (real, optional)
    R.bin, K.bin                        required
    rho.bin                             optional: the stationary density (recomputed if absent)
    gradient.bin                        optional: the gradient at `energy`

    Every .bin file is a headerless D x D complex128 matrix, little-endian, column-major
    (what Julia's write(io, Matrix{ComplexF64}) writes on a little-endian machine), of
    exactly 16 D^2 bytes. Readers ignore other keys and files; writers may add keys, and
    .bin files that are D x D complex128 matrices too.

read_portable also reads the unversioned directories of the research code (metadata.toml
without `format`): initial/ and optimized_state/ of the GPU optimizer before format 1, and
the Julia exports of the research tree. They describe phi4 (mass 1) and are normalized so:
    integration      n_quad from `quadrature` or `n_quad`; max_point from `window`,
                     `half_window` or `max_point`; atol, rtol (and maxiters) as such; none
                     if neither n_quad nor max_point is given. When the energy, gradient or
                     gradient norm comes from the saved_* keys (below) and a
                     [saved_integration] table is present, that table instead: it holds the
                     settings of those values, the top-level keys those of the GPU run.
    energy           `energy`, else `saved_energy` unless saved_energy_available = false
    gradient         gradient.bin, else saved_gradient.bin unless
                     saved_gradient_available = false
    gradient_norm    `gradient_norm`, else `saved_gradient_norm` unless
                     saved_gradient_norm_available (default: saved_gradient_available)
                     is false
    manifold         top-level metric_reg and precond_reg, when both are given
    validation_passed, source   as such
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import tomllib

import numpy as np

__all__ = ["FORMAT", "FORMAT_VERSION", "Integration", "Manifold", "StateRecord",
           "read_jld2", "read_portable", "write_portable"]

FORMAT = "rcmps-portable-state"
FORMAT_VERSION = 1
# The keys of format 1, in the order write_portable writes them.
KEYS = ("format", "format_version", "D", "model", "coupling", "mass", "energy", "gradient_norm",
        "validation_passed", "producer", "source", "info", "integration", "manifold")
MATRICES = ("R", "K", "rho", "gradient")


@dataclass(frozen=True)
class Integration:
    """IntegratorParameters of RcmpsCore: `max_point` is the physical half-window (mass one),
    `n_quad` the Gauss nodes per half."""
    n_quad: int
    max_point: float
    atol: float
    rtol: float
    maxiters: int | None = None


@dataclass(frozen=True)
class Manifold:
    """ManifoldParameters of RcmpsCore."""
    metric_reg: float
    precond_reg: float
    precond_reg_prefactor: float | None = None


@dataclass(frozen=True, eq=False)
class StateRecord:
    """A state as stored. Matrices are C-ordered complex128 (D, D) arrays, or None when
    absent (and R, K when read with arrays=False); energy, gradient_norm,
    validation_passed, integration, manifold, producer, source and info are None when
    absent. `origin` is "jld2", "portable" (format 1) or "portable-legacy"; `files`
    lists the files of a portable directory that the record was read from, and
    `metadata` holds its metadata.toml as parsed (with the keys readers ignore)."""
    D: int
    model: str
    coupling: float
    R: np.ndarray | None = None
    K: np.ndarray | None = None
    mass: float = 1.0
    rho: np.ndarray | None = None
    gradient: np.ndarray | None = None
    energy: float | None = None
    gradient_norm: float | None = None
    validation_passed: bool | None = None
    integration: Integration | None = None
    manifold: Manifold | None = None
    producer: str | None = None
    source: str | None = None
    info: str | None = None
    origin: str = ""
    files: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- checks

def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, not {value!r}")
    return int(value)


def _real(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number, not {value!r}")
    value = float(value)
    if not math.isfinite(value) or positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"{name} must be finite{' and positive' if positive else ''}"
                         f"{' and nonnegative' if nonnegative else ''}, not {value!r}")
    return value


def _text(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, not {value!r}")
    return value


def _flag(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be true or false, not {value!r}")
    return bool(value)


def _optional(convert, value, name, **options):
    return None if value is None else convert(value, name, **options)


def _check_model(model, mass):
    if not model:
        raise ValueError("the model name is empty")
    if model == "phi4" and mass != 1.0:
        raise ValueError(f"model phi4 has no mass parameter (its normal-ordering mass is 1): "
                         f"this state has mass = {mass!r}, refused")


def _integration(table, name):
    if not isinstance(table, dict):
        raise ValueError(f"{name} must be a table")
    try:
        return Integration(_integer(table["n_quad"], f"{name}.n_quad"),
                           _real(table["max_point"], f"{name}.max_point", positive=True),
                           _real(table["atol"], f"{name}.atol", positive=True),
                           _real(table["rtol"], f"{name}.rtol", positive=True),
                           _optional(_integer, table.get("maxiters"), f"{name}.maxiters"))
    except KeyError as error:
        raise ValueError(f"{name} has no {error.args[0]}") from None


def _manifold(table, name):
    if not isinstance(table, dict):
        raise ValueError(f"{name} must be a table")
    try:
        return Manifold(_real(table["metric_reg"], f"{name}.metric_reg", nonnegative=True),
                        _real(table["precond_reg"], f"{name}.precond_reg", nonnegative=True),
                        _optional(_real, table.get("precond_reg_prefactor"), f"{name}.precond_reg_prefactor",
                                  nonnegative=True))
    except KeyError as error:
        raise ValueError(f"{name} has no {error.args[0]}") from None


def _matrix(value, name, d):
    """A finite (d, d) matrix as a C-ordered native complex128 array."""
    x = np.asarray(value)
    if x.shape != (d, d) or not np.issubdtype(x.dtype, np.number):
        raise ValueError(f"{name} must be a {d}x{d} matrix, not of shape {x.shape} and type {x.dtype}")
    x = np.ascontiguousarray(x, dtype=np.complex128)
    if not np.isfinite(x).all():
        raise ValueError(f"{name} holds nonfinite values")
    return x


# ----------------------------------------------------------------------------- JLD2

def _complex(dataset):
    value = dataset[()]
    out = np.empty(np.shape(value), dtype=np.complex128)
    out.real, out.imag = value["re"], value["im"]   # bit for bit, signed zeros included
    return out


def _string(dataset):
    value = dataset[()]
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


def read_jld2(path):
    """Read an rcmps_store checkpoint (JLD2 schema 1) with h5py. rho is not stored (None);
    gradient and energy are None when e0 is not finite (a state never evaluated)."""
    import h5py

    with h5py.File(path, "r") as f:
        if int(f["schema_version"][()]) != 1:
            raise ValueError("unsupported schema; migrate old JLD with examples/migrate_legacy.jl")
        R, K = _complex(f["R"]).T, _complex(f["K"]).T
        coupling = _complex(f["coupling"])
        model = _string(f["model"])
        mass = float(f["mass"][()]) if "mass" in f else 1.0
        e0 = float(f["e0"][()])
        grad = _complex(f["grad"]).T if "grad" in f else None
        try:
            info = _string(f["info"]) if "info" in f else None
        except (KeyError, OSError):   # JLD2 writes "" with a datatype of size zero, which HDF5 refuses
            info = ""
        row = f["integration"][()]
        integration = Integration(int(row["n_quad"]), float(row["max_point"]), float(row["atol"]),
                                  float(row["rtol"]),
                                  int(row["maxiters"]) if "maxiters" in row.dtype.names else None)
        row = f["manifold"][()] if "manifold" in f else None
        manifold = None if row is None else Manifold(
            float(row["metric_reg"]), float(row["precond_reg"]),
            float(row["precond_reg_prefactor"]) if "precond_reg_prefactor" in row.dtype.names else None)
    if coupling.imag != 0:
        raise ValueError("model coupling must be real")
    _check_model(model, mass)
    if R.ndim != 2 or R.shape[0] != R.shape[1] or R.shape[0] == 0:
        raise ValueError("R must be nonempty and square")
    d = R.shape[0]
    evaluated = math.isfinite(e0)
    return StateRecord(D=d, model=model, coupling=float(coupling.real), R=_matrix(R, "R", d),
                       K=_matrix(K, "K", d), mass=mass,
                       gradient=_matrix(grad, "grad", d) if evaluated and grad is not None else None,
                       energy=e0 if evaluated else None, integration=integration, manifold=manifold,
                       info=info, origin="jld2")


# ----------------------------------------------------------------------------- portable

def _format1(metadata):
    if metadata["format"] != FORMAT:
        raise ValueError(f"metadata.toml: format {metadata['format']!r} is not {FORMAT!r}")
    version = metadata.get("format_version")
    if isinstance(version, bool) or version != FORMAT_VERSION:
        raise ValueError(f"metadata.toml: unsupported {FORMAT} format_version {version!r} "
                         f"(this reader knows {FORMAT_VERSION})")
    for key in ("D", "model", "coupling"):
        if key not in metadata:
            raise ValueError(f"metadata.toml: required key {key} is missing")
    get = metadata.get
    return dict(
        D=_integer(get("D"), "D"), model=_text(get("model"), "model"),
        coupling=_real(get("coupling"), "coupling"), mass=_real(get("mass", 1.0), "mass"),
        energy=_optional(_real, get("energy"), "energy"),
        gradient_norm=_optional(_real, get("gradient_norm"), "gradient_norm", nonnegative=True),
        validation_passed=_optional(_flag, get("validation_passed"), "validation_passed"),
        producer=_optional(_text, get("producer"), "producer"),
        source=_optional(_text, get("source"), "source"),
        info=_optional(_text, get("info"), "info"),
        integration=_optional(_integration, get("integration"), "[integration]"),
        manifold=_optional(_manifold, get("manifold"), "[manifold]"),
        origin="portable"), "gradient.bin"


def _legacy(metadata, directory):
    get = metadata.get
    for key in ("D", "coupling"):
        if key not in metadata:
            raise ValueError(f"metadata.toml: required key {key} is missing")
    saved = False   # whether a value comes from the saved_* keys

    def available(flag, default=True):
        return get(flag, default) is not False

    energy = get("energy")
    if energy is None and "saved_energy" in metadata and available("saved_energy_available"):
        energy, saved = get("saved_energy"), True
    gradient = "gradient.bin"
    if not (directory / gradient).exists():
        gradient = "saved_gradient.bin"
        if (directory / gradient).exists() and available("saved_gradient_available"):
            saved = True
        else:
            gradient = None
    norm = get("gradient_norm")
    if norm is None and "saved_gradient_norm" in metadata and available(
            "saved_gradient_norm_available", available("saved_gradient_available")):
        norm, saved = get("saved_gradient_norm"), True
    if saved and "saved_integration" in metadata:
        integration = _integration(get("saved_integration"), "[saved_integration]")
    else:
        n_quad = get("quadrature", get("n_quad"))
        max_point = get("window", get("half_window", get("max_point")))
        integration = None if n_quad is None and max_point is None else _integration(
            dict(n_quad=n_quad, max_point=max_point, atol=get("atol"), rtol=get("rtol"), maxiters=get("maxiters")),
            "metadata.toml (integration keys)")
    manifold = (_manifold(metadata, "metadata.toml (metric_reg, precond_reg)")
                if "metric_reg" in metadata and "precond_reg" in metadata else None)
    return dict(
        D=_integer(get("D"), "D"), model="phi4", coupling=_real(get("coupling"), "coupling"), mass=1.0,
        energy=_optional(_real, energy, "energy"),
        gradient_norm=_optional(_real, norm, "gradient_norm", nonnegative=True),
        validation_passed=_optional(_flag, get("validation_passed"), "validation_passed"),
        source=_optional(_text, get("source"), "source"),
        integration=integration, manifold=manifold, origin="portable-legacy"), gradient


def _read_bin(path, d):
    size = path.stat().st_size
    if size != 16 * d * d:
        raise ValueError(f"{path.name} has {size} bytes, not the {16 * d * d} of a {d}x{d} complex128 matrix")
    return _matrix(np.fromfile(path, dtype="<c16").reshape((d, d), order="F"), path.name, d)


def read_portable(directory, *, arrays=True):
    """Read a portable state directory: format 1 or a research dialect (module docstring).
    With arrays=False only metadata.toml is read and checked (the matrices are None, and
    `files` names the files a full read would read)."""
    directory = Path(directory)
    with (directory / "metadata.toml").open("rb") as stream:
        metadata = tomllib.load(stream)
    fields, gradient = _format1(metadata) if "format" in metadata else _legacy(metadata, directory)
    _check_model(fields["model"], fields["mass"])
    names = dict(R="R.bin", K="K.bin", rho="rho.bin", gradient=gradient)
    names = {key: name for key, name in names.items()
             if name is not None and (key in ("R", "K") or (directory / name).exists())}
    files = ("metadata.toml", *names.values())
    if arrays:
        fields.update({key: _read_bin(directory / name, fields["D"]) for key, name in names.items()})
    return StateRecord(**fields, files=files, metadata=metadata)


def _toml_string(text):
    escapes = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}
    return '"' + "".join(escapes.get(c, f"\\u{ord(c):04X}" if ord(c) < 0x20 or ord(c) == 0x7F else c)
                         for c in text) + '"'


def _toml_value(value, name):
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(_real(value, name))   # shortest repr: reads back exactly
    if isinstance(value, str):
        return _toml_string(value)
    raise TypeError(f"{name}: a metadata value must be a string, boolean, integer or real, not {value!r}")


def _table(name, values):
    rows = [f"{key} = {_toml_value(value, f'{name}.{key}')}" for key, value in values.items() if value is not None]
    return f"\n[{name}]\n" + "".join(row + "\n" for row in rows)


def write_portable(directory, record, *, extra=None, extra_matrices=None):
    """Write `record` as a format-1 directory, which must not exist (an existing one, even
    empty, is refused). `extra` adds top-level keys (strings, booleans, integers, reals)
    and `extra_matrices` .bin files ({name: D x D matrix}); neither may reuse a name of
    the format. The .bin files are written first and metadata.toml last, by a rename.
    record.origin, files and metadata are not written. Returns the directory."""
    directory = Path(directory)
    d = _integer(record.D, "D")
    if record.R is None or record.K is None:
        raise ValueError("R and K are required")
    matrices = {name: _matrix(getattr(record, name), name, d) for name in MATRICES
                if getattr(record, name) is not None}
    for name, matrix in (extra_matrices or {}).items():
        if name in MATRICES or not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError(f"extra matrix name {name!r} is not allowed")
        matrices[name] = _matrix(matrix, name, d)
    model, mass = _text(record.model, "model"), _real(record.mass, "mass")
    _check_model(model, mass)
    head = dict(format=FORMAT, format_version=FORMAT_VERSION, D=d, model=model,
                coupling=_real(record.coupling, "coupling"), mass=mass,
                energy=_optional(_real, record.energy, "energy"),
                gradient_norm=_optional(_real, record.gradient_norm, "gradient_norm", nonnegative=True),
                validation_passed=_optional(_flag, record.validation_passed, "validation_passed"),
                producer=_optional(_text, record.producer, "producer"),
                source=_optional(_text, record.source, "source"), info=_optional(_text, record.info, "info"))
    for key in extra or {}:
        if key in KEYS or not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ValueError(f"extra metadata key {key!r} is not allowed")
    text = "".join(f"{key} = {_toml_value(value, key)}\n" for key, value in {**head, **(extra or {})}.items()
                   if value is not None)
    if record.integration is not None:
        i = record.integration
        text += _table("integration", dict(
            n_quad=_integer(i.n_quad, "integration.n_quad"),
            max_point=_real(i.max_point, "integration.max_point", positive=True),
            atol=_real(i.atol, "integration.atol", positive=True), rtol=_real(i.rtol, "integration.rtol", positive=True),
            maxiters=_optional(_integer, i.maxiters, "integration.maxiters")))
    if record.manifold is not None:
        m = record.manifold
        text += _table("manifold", dict(
            metric_reg=_real(m.metric_reg, "manifold.metric_reg", nonnegative=True),
            precond_reg=_real(m.precond_reg, "manifold.precond_reg", nonnegative=True),
            precond_reg_prefactor=_optional(_real, m.precond_reg_prefactor, "manifold.precond_reg_prefactor",
                                            nonnegative=True)))
    tomllib.loads(text)   # a writer bug must not produce an unreadable file
    directory.mkdir(parents=True, exist_ok=False)
    for name, matrix in matrices.items():
        matrix.ravel(order="F").astype("<c16").tofile(directory / f"{name}.bin")
    temporary = directory / ".metadata.toml.tmp"
    temporary.write_bytes(text.encode("utf-8"))
    os.replace(temporary, directory / "metadata.toml")
    return directory
