# Portable state directories, "rcmps-portable-state" format 1, on the Julia side: the
# format is specified in gpu/src/rcmps_gpu/io.py (module docstring), which also reads the
# unversioned research directories in the same way as read_portable here.
# Included by the scripts of this directory; uses the RcmpsCore 0.2 public API and the
# LinearAlgebra and TOML standard libraries only.
using RcmpsCore, LinearAlgebra, TOML

const PORTABLE_FORMAT = "rcmps-portable-state"
const PORTABLE_FORMAT_VERSION = 1
# Order of the format-1 keys in metadata.toml; other keys follow, sorted.
const PORTABLE_KEYS = ["format","format_version","D","model","coupling","mass","symmetry","energy","gradient_norm",
                       "phi","validation_passed","producer","source","info"]
# Values of the optional key symmetry (added to format 1 after rcmps_gpu 0.1.0).
const PORTABLE_SYMMETRIES = ("Z2", "none")

Base.ENDIAN_BOM == 0x04030201 || error("portable state files are little-endian; this machine is big-endian")

"""Refuse what rcmps_gpu.io refuses too: an empty model name, and a phi4 mass other than 1."""
function portable_check_model(name, mass)
    isempty(string(name)) && error("the model name is empty")
    name == "phi4" && mass != 1.0 &&
        error("model phi4 has no mass parameter (its normal-ordering mass is 1): this state has mass = $mass, refused")
    return nothing
end

"""The ModelParameters of a state; a mass other than 1.0 needs an RcmpsCore that has one."""
function portable_model(name::AbstractString, coupling::Real, mass::Real)
    portable_check_model(name, mass)
    hasfield(ModelParameters, :mass) && return ModelParameters(name=name, coupling=coupling, mass=mass)
    mass == 1.0 || error("this RcmpsCore has no model mass; the state has mass = $mass")
    return ModelParameters(name=name, coupling=coupling)
end
portable_mass(m::ModelParameters) = hasfield(ModelParameters, :mass) ? Float64(getfield(m, :mass)) : 1.0

function read_portable_matrix(path, d)
    size = filesize(path)
    size == 16d^2 || error("$(basename(path)) has $size bytes, not the $(16d^2) of a $(d)x$(d) complex128 matrix")
    x = Matrix{ComplexF64}(reshape(reinterpret(ComplexF64, read(path)), d, d))
    all(isfinite, x) || error("$(basename(path)) holds nonfinite values")
    return x
end

function portable_integer(value, name; minimum=1)
    value isa Integer && !(value isa Bool) && value >= minimum || error("$name must be an integer >= $minimum, not $(repr(value))")
    return Int(value)
end
function portable_real(value, name; positive=false, nonnegative=false)
    value isa Real && !(value isa Bool) || error("$name must be a real number, not $(repr(value))")
    x = Float64(value)
    isfinite(x) && !(positive && x <= 0) && !(nonnegative && x < 0) ||
        error("$name must be finite$(positive ? " and positive" : "")$(nonnegative ? " and nonnegative" : ""), not $x")
    return x
end
portable_optional(f, value, name; options...) = isnothing(value) ? nothing : f(value, name; options...)
function portable_text(value, name)
    value isa AbstractString || error("$name must be a string, not $(repr(value))")
    return String(value)
end
function portable_symmetry(value, name)
    value in PORTABLE_SYMMETRIES || error("$name must be one of $(join(repr.(PORTABLE_SYMMETRIES), ", ")), not $(repr(value))")
    return String(value)
end
function portable_flag(value, name)
    value isa Bool || error("$name must be true or false, not $(repr(value))")
    return value
end
function portable_integration(table, name)
    table isa AbstractDict || error("$name must be a table")
    for key in ("n_quad","max_point","atol","rtol")
        haskey(table, key) && !isnothing(table[key]) || error("$name has no $key")
    end
    return (n_quad=portable_integer(table["n_quad"], "$name.n_quad"),
            max_point=portable_real(table["max_point"], "$name.max_point"; positive=true),
            atol=portable_real(table["atol"], "$name.atol"; positive=true),
            rtol=portable_real(table["rtol"], "$name.rtol"; positive=true),
            maxiters=portable_optional(portable_integer, get(table, "maxiters", nothing), "$name.maxiters"))
end
function portable_manifold(table, name)
    table isa AbstractDict || error("$name must be a table")
    for key in ("metric_reg","precond_reg")
        haskey(table, key) || error("$name has no $key")
    end
    return (metric_reg=portable_real(table["metric_reg"], "$name.metric_reg"; nonnegative=true),
            precond_reg=portable_real(table["precond_reg"], "$name.precond_reg"; nonnegative=true),
            precond_reg_prefactor=portable_optional(portable_real, get(table, "precond_reg_prefactor", nothing),
                                                    "$name.precond_reg_prefactor"; nonnegative=true))
end

# Legacy (research) metadata: a saved_* value is available unless its flag is false; the first key present.
portable_available(metadata, flag, default=true) = get(metadata, flag, default) !== false
portable_first(metadata, keys...) = something((get(metadata, key, nothing) for key in keys)..., Some(nothing))

"""Read a portable state directory (format 1, or an unversioned research directory, normalized
as rcmps_gpu.io.read_portable does). Absent values are `nothing`; `metadata` is the parsed TOML."""
function read_portable(directory::AbstractString)
    metadata = TOML.parsefile(joinpath(directory, "metadata.toml"))
    for key in ("D","coupling")
        haskey(metadata, key) || error("metadata.toml: required key $key is missing")
    end
    has(name) = isfile(joinpath(directory, name))
    if haskey(metadata, "format")
        metadata["format"] == PORTABLE_FORMAT || error("metadata.toml: format $(repr(metadata["format"])) is not $PORTABLE_FORMAT")
        version = get(metadata, "format_version", nothing)
        version isa Integer && !(version isa Bool) && version == PORTABLE_FORMAT_VERSION ||
            error("metadata.toml: unsupported $PORTABLE_FORMAT format_version $(repr(version)) (this reader knows $PORTABLE_FORMAT_VERSION)")
        haskey(metadata, "model") || error("metadata.toml: required key model is missing")
        model = portable_text(metadata["model"], "model")
        mass = portable_real(get(metadata, "mass", 1.0), "mass")
        energy, gnorm = get(metadata, "energy", nothing), get(metadata, "gradient_norm", nothing)
        gradient = has("gradient.bin") ? "gradient.bin" : nothing
        integration = portable_optional(portable_integration, get(metadata, "integration", nothing), "[integration]")
        manifold = portable_optional(portable_manifold, get(metadata, "manifold", nothing), "[manifold]")
        producer = portable_optional(portable_text, get(metadata, "producer", nothing), "producer")
        info = portable_optional(portable_text, get(metadata, "info", nothing), "info")
        symmetry = portable_optional(portable_symmetry, get(metadata, "symmetry", nothing), "symmetry")
        phi = portable_optional(portable_real, get(metadata, "phi", nothing), "phi")
        legacy = false
    else
        model, mass, producer, info, legacy = "phi4", 1.0, nothing, nothing, true
        symmetry, phi = nothing, nothing
        saved = false   # whether a value comes from the saved_* keys
        energy = get(metadata, "energy", nothing)
        if isnothing(energy) && haskey(metadata, "saved_energy") && portable_available(metadata, "saved_energy_available")
            energy, saved = metadata["saved_energy"], true
        end
        gradient = "gradient.bin"
        if !has(gradient)
            gradient = "saved_gradient.bin"
            if has(gradient) && portable_available(metadata, "saved_gradient_available")
                saved = true
            else
                gradient = nothing
            end
        end
        gnorm = get(metadata, "gradient_norm", nothing)
        if isnothing(gnorm) && haskey(metadata, "saved_gradient_norm") &&
                portable_available(metadata, "saved_gradient_norm_available",
                                   portable_available(metadata, "saved_gradient_available"))
            gnorm, saved = metadata["saved_gradient_norm"], true
        end
        if saved   # the top-level keys are the GPU run's settings, not those of the saved values
            integration = haskey(metadata, "saved_integration") ?
                portable_integration(metadata["saved_integration"], "[saved_integration]") : nothing
        else
            n_quad = portable_first(metadata, "quadrature", "n_quad")
            max_point = portable_first(metadata, "window", "half_window", "max_point")
            integration = isnothing(n_quad) && isnothing(max_point) ? nothing : portable_integration(
                Dict("n_quad"=>n_quad, "max_point"=>max_point, "atol"=>get(metadata, "atol", nothing),
                     "rtol"=>get(metadata, "rtol", nothing), "maxiters"=>get(metadata, "maxiters", nothing)),
                "metadata.toml (integration keys)")
        end
        manifold = haskey(metadata, "metric_reg") && haskey(metadata, "precond_reg") ?
            portable_manifold(metadata, "metadata.toml (metric_reg, precond_reg)") : nothing
    end
    portable_check_model(model, mass)
    d = portable_integer(metadata["D"], "D")
    matrix(name) = read_portable_matrix(joinpath(directory, name), d)
    return (D=d, model=model, coupling=portable_real(metadata["coupling"], "coupling"), mass=mass,
            R=matrix("R.bin"), K=matrix("K.bin"), rho=has("rho.bin") ? matrix("rho.bin") : nothing,
            gradient=isnothing(gradient) ? nothing : matrix(gradient),
            energy=portable_optional(portable_real, energy, "energy"),
            gradient_norm=portable_optional(portable_real, gnorm, "gradient_norm"; nonnegative=true),
            symmetry=symmetry, phi=phi,
            validation_passed=portable_optional(portable_flag, get(metadata, "validation_passed", nothing), "validation_passed"),
            integration=integration, manifold=manifold, producer=producer,
            source=portable_optional(portable_text, get(metadata, "source", nothing), "source"), info=info,
            legacy=legacy, metadata=metadata)
end

"""Write a format-1 directory, which must not exist. `R`, `K`, `rho`, `gradient`: matrices or
`nothing`; `integration`, `manifold`: NamedTuples (fields as in the format) or `nothing`;
`symmetry`: "Z2", "none" or `nothing`; `extra`: further top-level keys. The .bin files are
written first, metadata.toml last."""
function write_portable(directory::AbstractString; R, K, model, coupling, mass=1.0, rho=nothing, gradient=nothing,
                        energy=nothing, gradient_norm=nothing, symmetry=nothing, phi=nothing, validation_passed=nothing,
                        producer=nothing, source=nothing, info=nothing, integration=nothing, manifold=nothing,
                        extra=Dict{String,Any}())
    ispath(directory) && error("Refusing to overwrite $directory: choose a new directory")
    d = size(R, 1)
    portable_check_model(model, mass)
    imag(coupling) == 0 || error("model coupling must be real")
    metadata = Dict{String,Any}("format"=>PORTABLE_FORMAT, "format_version"=>PORTABLE_FORMAT_VERSION, "D"=>d,
        "model"=>String(model), "coupling"=>portable_real(real(coupling), "coupling"), "mass"=>portable_real(mass, "mass"))
    isnothing(symmetry) || (metadata["symmetry"] = portable_symmetry(symmetry, "symmetry"))
    for (key, value) in (("energy", energy), ("gradient_norm", gradient_norm), ("phi", phi),
                         ("validation_passed", validation_passed), ("producer", producer), ("source", source),
                         ("info", info))
        isnothing(value) || (metadata[key] = value isa Real && !(value isa Bool) ? portable_real(value, key) : value)
    end
    for (key, value) in extra
        (key in PORTABLE_KEYS || key in ("integration","manifold")) && error("extra key $key is a key of the format")
        metadata[key] = value
    end
    for (name, table) in (("integration", integration), ("manifold", manifold))
        isnothing(table) || (metadata[name] = Dict{String,Any}(string(k)=>v for (k, v) in pairs(table) if !isnothing(v)))
    end
    matrices = [(name, x) for (name, x) in (("R", R), ("K", K), ("rho", rho), ("gradient", gradient)) if !isnothing(x)]
    for (name, x) in matrices
        size(x) == (d, d) && all(isfinite, x) || error("$name must be a finite $(d)x$(d) matrix")
    end
    mkpath(dirname(abspath(directory)))
    mkdir(directory)
    for (name, x) in matrices
        open(joinpath(directory, name * ".bin"), "w") do io
            write(io, Matrix{ComplexF64}(x))   # column-major, native (little-endian) byte order
        end
    end
    temporary = joinpath(directory, ".metadata.toml.tmp")
    open(temporary, "w") do io
        TOML.print(io, metadata; sorted=true,
                   by=key -> (something(findfirst(==(key), PORTABLE_KEYS), length(PORTABLE_KEYS) + 1), key))
    end
    mv(temporary, joinpath(directory, "metadata.toml"))
    return directory
end
