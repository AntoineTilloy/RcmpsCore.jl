"""Strict NumPy-backed stand-in for CuPy: the host/device failure classes that NumPy hides.

There is no GPU on the development machine, and with xp = numpy every host/device
mixing bug of the xp-generic modules (scaled, kinetic, potential, ode, hamiltonian,
bilinear, eigsolve, packed) is silent. `import fakecupy as cp` gives a module with
cp.__name__ == 'cupy' (never registered as `cupy` in sys.modules) whose arrays are
NumPy arrays behind a wrapper, `cp.ndarray`, with CuPy's rules for exactly those classes:

* arithmetic, matmul, comparisons, in-place operators and item assignment raise
  TypeError for a numpy.ndarray operand; Python and NumPy scalars are accepted;
* NumPy cannot convert a device array (numpy.asarray, numpy.array, a device array as
  index of a NumPy array): only .get() and asnumpy download, and they copy;
* module functions take device arrays and scalars; only asarray and array upload a
  host array, and they copy, so host and device memory never alias;
* reductions and elements are 0-d device arrays, never NumPy scalars, and an element
  is a view (y = x[i]; y += 1 writes x, as in CuPy); float(), complex(), int(), bool()
  of a 0-d array work; .conj() and .real of a real array return the array itself;
* a device array is indexed by ints, slices, device arrays, and NumPy integer or
  boolean arrays and lists (CuPy uploads them);
* RawKernel, RawModule, ElementwiseKernel and cuda.* raise NotImplementedError.

Deliberately STRICTER than CuPy (a pass here is safe there; a failure here may be legal):
NumPy functions and ufuncs on a device array raise (CuPy dispatches them to itself by
__array_ufunc__ / __array_function__); 0-d numpy.ndarray operands and values are
refused (CuPy takes them in comparisons and assignments); lists and tuples are refused
as operands and array arguments, and asarray of a sequence holding device arrays is
refused (CuPy stacks it); concatenate and stack need a list or tuple of device arrays;
the fill value of `full` is a scalar; float(), int(), complex() need ndim 0, there is no
__index__, and a format specification is refused (f"{x:.3e}" needs float(x)); only the
functions, methods and dtypes listed here exist (linalg has norm only); empty and
empty_like return NaN, since a GPU memory pool hands back stale data where NumPy
usually returns zeros; ReductionKernel needs the declared dtypes exactly.

LOOSER than CuPy, i.e. not checked here: everything is computed by NumPy, so indexing
corner cases, ufunc keywords, dtype promotion and out-of-bounds indices (CuPy wraps
them silently) follow NumPy 2; overlapping in-place operations and repeated indices in
an assignment are well defined here and not on a GPU; ReductionKernel is evaluated
for sum reductions whose map expression is also a NumPy expression (the Gram-Schmidt
kernels of scaled.projections), so its CUDA text is neither compiled nor checked.
Nothing of CUDA is emulated: no streams, memory pools, devices or kernels.
"""
from __future__ import annotations

import builtins as _py
import operator as _op

import numpy as _np

_SCALARS = (bool, int, float, complex, _np.generic)
_DTYPES = frozenset("?bhilqBHILQefdFD")      # bool, (u)int8-64, float16-64, complex64-128
_CONVERSION = ("Implicit conversion to a NumPy array is not allowed. "
               "Please use `.get()` to construct a NumPy array explicitly.")   # CuPy's message


def _refuse(where, x):
    raise TypeError(f"fakecupy {where}: unsupported type {type(x).__module__}.{type(x).__qualname__}; "
                    "device arrays and Python or NumPy scalars only (upload with asarray, download with .get())")


def _operand(x, where):
    """Backing array of a device array, or a Python/NumPy scalar; anything else is refused."""
    if isinstance(x, ndarray):
        return x._a
    if isinstance(x, _SCALARS):
        return x
    _refuse(where, x)


def _plain(x, where):
    """Any other argument (shape, axis, dtype, out, ...): device arrays are unwrapped,
    also inside lists and tuples; NumPy arrays are refused."""
    if isinstance(x, ndarray):
        return x._a
    if isinstance(x, _np.ndarray):
        _refuse(where, x)
    if isinstance(x, (list, tuple)):
        return type(x)(_plain(v, where) for v in x)
    return x


def _wrap(a):
    """Device array on the NumPy array `a`, without a copy; NumPy scalars become 0-d arrays."""
    a = _np.asarray(a)
    if a.dtype.char not in _DTYPES:
        raise TypeError(f"fakecupy: unsupported dtype {a.dtype}")
    out = object.__new__(ndarray)
    out._a = a
    return out


def _index(key):
    """Index of the backing array. Device arrays are unwrapped; NumPy integer or boolean
    arrays and lists pass, as CuPy uploads them."""
    out = []
    for k in key if isinstance(key, tuple) else (key,):
        if isinstance(k, ndarray):
            k = k._a
        if isinstance(k, _np.ndarray) and k.dtype.kind not in "iub":
            raise IndexError("fakecupy: arrays used as indices must be of integer or boolean type")
        out.append(k)
    return tuple(out)


class _Flags:
    def __init__(self, a):
        self.c_contiguous, self.f_contiguous, self.owndata = a.flags.c_contiguous, a.flags.f_contiguous, a.flags.owndata


class ndarray:
    """Device array; `_a` is the backing NumPy array and is private to this module."""
    __slots__ = ("_a",)
    __array_ufunc__ = None     # NumPy operators defer to the reflected methods; NumPy ufuncs raise
    __hash__ = None

    def __init__(self, *args, **kw):
        raise NotImplementedError("fakecupy.ndarray: create device arrays with asarray, zeros, ...")

    # ---- what NumPy must not do --------------------------------------------------------
    def __array__(self, *args, **kw):
        raise TypeError(_CONVERSION)

    def __array_function__(self, function, types, args, kw):
        raise TypeError(f"fakecupy: numpy.{function.__name__} called on a device array; use the device "
                        "module (xp) or an explicit .get() copy")

    def __getattr__(self, name):
        raise AttributeError(f"fakecupy.ndarray has no attribute {name!r} (not provided by the strict stand-in)")

    # ---- scalars and text ---------------------------------------------------------------
    def _item(self, kind):
        if self._a.ndim:
            raise TypeError(f"fakecupy: {kind}() needs a 0-d device array, not shape {self._a.shape}")
        return self._a[()]

    def __float__(self):
        return float(self._item("float"))

    def __int__(self):
        return int(self._item("int"))

    def __complex__(self):
        return complex(self._item("complex"))

    def __bool__(self):
        return bool(self._a)

    def __repr__(self):
        return repr(self._a)

    def __str__(self):
        return str(self._a)

    def __format__(self, spec):
        if spec:
            raise TypeError("fakecupy: format specification on a device array; convert with float() or complex()")
        return str(self._a)

    # ---- container -----------------------------------------------------------------------
    def __len__(self):
        return len(self._a)

    def __iter__(self):
        if not self._a.ndim:
            raise TypeError("iteration over a 0-d array")
        return (self[i] for i in range(len(self._a)))

    def __getitem__(self, key):
        key = _index(key)
        if not _py.any(k is Ellipsis for k in key):
            key += (Ellipsis,)       # an element is a 0-d view, not a scalar
        return _wrap(self._a[key])

    def __setitem__(self, key, value):
        self._a[_index(key)] = _operand(value, "item assignment")

    # ---- attributes -------------------------------------------------------------------------
    shape = property(lambda self: self._a.shape)
    ndim = property(lambda self: self._a.ndim)
    dtype = property(lambda self: self._a.dtype)
    size = property(lambda self: self._a.size)
    nbytes = property(lambda self: self._a.nbytes)
    itemsize = property(lambda self: self._a.itemsize)
    strides = property(lambda self: self._a.strides)
    flags = property(lambda self: _Flags(self._a))
    T = property(lambda self: _wrap(self._a.T))

    @property
    def real(self):
        return _wrap(self._a.real) if self._a.dtype.kind == "c" else self

    @real.setter
    def real(self, value):
        self._a.real = _operand(value, "real")

    @property
    def imag(self):
        return _wrap(self._a.imag)

    @imag.setter
    def imag(self, value):
        self._a.imag = _operand(value, "imag")

    def conj(self):
        return _wrap(self._a.conj()) if self._a.dtype.kind == "c" else self

    conjugate = conj

    def get(self, stream=None, order="C", out=None):
        """Explicit download: a new NumPy array."""
        if stream is not None or out is not None:
            raise NotImplementedError("fakecupy: ndarray.get(stream=, out=)")
        return _np.array(self._a, order=order, copy=True)

    def item(self):
        return self._a.item()

    def tolist(self):
        return self._a.tolist()

    def fill(self, value):
        if not isinstance(value, _SCALARS):
            _refuse("fill", value)
        self._a.fill(value)


def _binary(name, function, inplace=None, reflect=True):
    def forward(self, other):
        return _wrap(function(self._a, _operand(other, name)))

    def reflected(self, other):
        return _wrap(function(_operand(other, name), self._a))

    def update(self, other):
        inplace(self._a, _operand(other, name))
        return self

    setattr(ndarray, f"__{name}__", forward)
    if reflect:
        setattr(ndarray, f"__r{name}__", reflected)
    if inplace is not None:
        setattr(ndarray, f"__i{name}__", update)


for _name in ("add", "sub", "mul", "truediv", "floordiv", "mod", "pow", "and", "or", "xor"):
    _binary(_name, getattr(_op, f"__{_name}__"), getattr(_op, f"__i{_name}__"))
_binary("matmul", _op.matmul)                           # a @= b rebinds
for _name in ("lt", "le", "gt", "ge", "eq", "ne"):      # Python reflects comparisons by itself
    _binary(_name, getattr(_op, _name), reflect=False)
for _name in ("neg", "pos", "abs", "invert"):
    setattr(ndarray, f"__{_name}__", (lambda f: lambda self: _wrap(f(self._a)))(getattr(_op, _name)))


def _method(name):
    def method(self, *args, **kw):
        return _wrap(getattr(self._a, name)(*(_plain(a, name) for a in args),
                                            **{k: _plain(v, name) for k, v in kw.items()}))
    method.__name__ = name
    return method


for _name in ("reshape", "copy", "astype", "sum", "all", "any", "max", "min", "mean", "trace", "swapaxes",
              "transpose", "ravel", "flatten", "squeeze"):
    setattr(ndarray, _name, _method(_name))


# ---- module functions ---------------------------------------------------------------------
def _poison(a):
    if a.dtype.kind in "fc":
        a.fill(complex(_np.nan, _np.nan) if a.dtype.kind == "c" else _np.nan)
    return a


def _function(name, arrays=1, source=_np, finish=lambda a: a):
    """source.name with its first `arrays` positional arguments device arrays or scalars."""
    function = getattr(source, name)

    def call(*args, **kw):
        if len(args) < arrays:
            raise TypeError(f"fakecupy.{name}: array arguments are positional")
        head = [_operand(a, name) for a in args[:arrays]]
        return _wrap(finish(function(*head, *(_plain(a, name) for a in args[arrays:]),
                                     **{k: _plain(v, name) for k, v in kw.items()})))
    call.__name__ = name
    return call


def _creation(name, finish=lambda a: a):
    function = getattr(_np, name)

    def call(*args, **kw):
        for value in (*args, *kw.values()):
            if isinstance(value, (ndarray, _np.ndarray)):
                _refuse(name, value)
        return _wrap(finish(function(*args, **kw)))
    call.__name__ = name
    return call


def _sequence(name):
    function = getattr(_np, name)

    def call(arrays, *args, **kw):
        if not isinstance(arrays, (list, tuple)):
            _refuse(name, arrays)
        for a in arrays:
            if not isinstance(a, ndarray):
                _refuse(name, a)        # CuPy: "Only cupy arrays can be concatenated"
        return _wrap(function([a._a for a in arrays], *(_plain(a, name) for a in args),
                              **{k: _plain(v, name) for k, v in kw.items()}))
    call.__name__ = name
    return call


def array(obj, dtype=None, copy=True, order="K", ndmin=0):
    """Explicit upload of a host array, sequence or scalar (always a copy), or a copy of a device array."""
    if isinstance(obj, ndarray):
        return _wrap(_np.array(obj._a, dtype=dtype, copy=True if copy else None, order=order, ndmin=ndmin))
    return _wrap(_np.array(obj, dtype=dtype, copy=True, order=order, ndmin=ndmin))


def asarray(a, dtype=None, order=None):
    """Explicit upload (a copy); a device array is returned as it is unless a cast is needed."""
    if isinstance(a, ndarray):
        b = _np.asarray(a._a, dtype=dtype, order=order)
        return a if b is a._a else _wrap(b)
    return _wrap(_np.array(a, dtype=dtype, order=order or "K", copy=True))


def asnumpy(a, stream=None, order="C", out=None):
    """Explicit download; a host array is returned as it is."""
    return a.get(stream, order, out) if isinstance(a, ndarray) else _np.asarray(a, order=order)


for _name in ("conj", "conjugate", "abs", "absolute", "sqrt", "exp", "real", "imag", "isfinite", "isnan", "sum",
              "amax", "amin", "max", "min", "all", "any", "trace", "diag", "moveaxis", "swapaxes", "transpose",
              "ascontiguousarray", "broadcast_to", "zeros_like", "ones_like", "copy", "reshape"):
    globals()[_name] = _function(_name)
for _name in ("maximum", "minimum", "matmul", "dot", "add", "subtract", "multiply", "divide"):
    globals()[_name] = _function(_name, 2)
where = _function("where", 3)
empty_like = _function("empty_like", finish=_poison)
for _name in ("zeros", "ones", "full", "eye", "arange"):
    globals()[_name] = _creation(_name)
empty = _creation("empty", _poison)
for _name in ("concatenate", "stack"):
    globals()[_name] = _sequence(_name)

complex128, complex64, float64, float32, int64, int32, bool_ = (
    _np.complex128, _np.complex64, _np.float64, _np.float32, _np.int64, _np.int32, _np.bool_)
dtype, pi, inf, nan, newaxis = _np.dtype, _np.pi, _np.inf, _np.nan, None


class _Namespace:
    def __init__(self, name, **functions):
        self.__dict__.update(functions, _name=name)

    def __getattr__(self, attribute):
        raise AttributeError(f"fakecupy.{self._name} has no attribute {attribute!r}: not provided by the strict "
                             "stand-in; do this algebra on the host with an explicit .get() copy")


linalg = _Namespace("linalg", norm=_function("norm", source=_np.linalg))


class ReductionKernel:
    """cupy.ReductionKernel for sum reductions only, the map expression evaluated by NumPy
    (operand names and conj). Arguments are device arrays of exactly the declared dtypes."""
    def __init__(self, in_params, out_params, map_expr, reduce_expr, post_map_expr, identity, name="reduce_kernel"):
        self.name = name
        self.inputs = [tuple(p.split()) for p in in_params.split(",")]
        out_type, out_name = out_params.split()
        if (reduce_expr.replace(" ", "") != "a+b" or post_map_expr.replace(" ", "") != f"{out_name}=a"
                or str(identity) != "0"):
            raise NotImplementedError("fakecupy.ReductionKernel: sum reductions only (a + b, out = a, identity 0)")
        self.dtype = _np.dtype(out_type)
        try:
            self.map = compile(map_expr, name, "eval")
        except SyntaxError as error:
            raise NotImplementedError("fakecupy.ReductionKernel: the map expression must also be a NumPy "
                                      "expression") from error

    def __call__(self, *args, axis=None, keepdims=False):
        if len(args) != len(self.inputs):
            raise TypeError(f"fakecupy.ReductionKernel {self.name}: {len(self.inputs)} arguments expected")
        scope = {}
        for (kind, operand), a in zip(self.inputs, args):
            if not isinstance(a, ndarray):
                _refuse(f"ReductionKernel {self.name}", a)
            if a.dtype != _np.dtype(kind):
                raise TypeError(f"fakecupy.ReductionKernel {self.name}: {operand} is {a.dtype}, declared {kind}")
            scope[operand] = a._a
        value = eval(self.map, {"__builtins__": {}, "conj": _np.conj}, scope)
        return _wrap(_np.sum(value, axis=axis, keepdims=keepdims, dtype=self.dtype))


class _NeedsGpu:
    """A CuPy name that exists but needs CUDA: calling it or reading an attribute raises."""
    def __init__(self, name):
        self.__dict__["_name"] = name

    def _fail(self, detail):
        raise NotImplementedError(f"fakecupy: cupy.{self._name}{detail} needs a real GPU; the strict stand-in is "
                                  "NumPy-backed and emulates no CUDA")

    def __call__(self, *args, **kw):
        self._fail("(...)")

    def __getattr__(self, attribute):
        if attribute.startswith("__"):
            raise AttributeError(attribute)
        self._fail("." + attribute)


RawKernel, RawModule, ElementwiseKernel, cuda, get_default_memory_pool, get_default_pinned_memory_pool = (
    _NeedsGpu(_name) for _name in ("RawKernel", "RawModule", "ElementwiseKernel", "cuda", "get_default_memory_pool",
                                   "get_default_pinned_memory_pool"))


def __getattr__(name):
    raise AttributeError(f"fakecupy has no attribute {name!r}: not provided by the strict stand-in "
                         "(add it here only if CuPy has it)")


del _name
__name__ = "cupy"      # code that inspects xp.__name__ takes its GPU branch
