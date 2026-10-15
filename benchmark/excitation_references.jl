# Reference excitation levels shared by RcmpsCore and the GPU package in gpu/ (rcmps_gpu):
# checks, or regenerates, test/fixtures/excitation_references.toml with the library entry point
# excitation_eigsolve.
#
# Run from the repository root (TOML, SHA and Dates are standard libraries):
#   julia --project=. --threads=2 benchmark/excitation_references.jl [--update] [--state NAME]...
#   julia --project=. benchmark/excitation_references.jl --write-state NAME NEW_FILE.jld2
#
# Check (the default). Every [[levels]] entry with regenerate = true is recomputed, or only those
# of the states named with --state. Entries of one solve (same state, settings, sector, momentum
# and nev; `level` is the index of the level in that solve) share one excitation_eigsolve call.
# Before the call, the state file's sha256 must equal [states.NAME].sha256 and its D, model and
# coupling the recorded ones. A level passes when the search converged and the level is accepted,
# its validated energy and each of its four validation energies (standard action, refined action,
# bilinear, refined bilinear) differ from the stored ones by at most `tolerance`, and its Ritz
# energy by at most `ritz_tolerance`. Exit status 0 when every level passes, 1 otherwise.
#
# --update recomputes the same entries and writes their computed fields (energies, residuals,
# operation counts, start, timing, date and versions) into the file. Every other entry and field
# is kept: the regenerate = false entries (the D=32 levels, the recorded mass-scan masses), the
# GPU values and the tolerances. A level whose recomputed results (energies, residual, spread,
# counts, start, status, dimension) all equal the stored ones bit for bit keeps its record, date
# and versions included; the file is written with TOML.print(...; sorted=true), so an update that
# changes no result rewrites it byte for byte. The differences are printed as in a check. If any
# recomputed level is not converged or not accepted, nothing is written and the status is 1.
#
# --write-state NAME FILE writes the state described by [states.NAME.recipe] (a RandomStateZ2)
# with rcmps_store into the new FILE. This is how test/fixtures/phi4_g1_D4_z2_state.jld2 was made;
# the committed file, checked by its sha256, remains the reference.
#
# Cost with two Julia threads and one BLAS thread: the D=4 levels take seconds after compilation,
# the three D=16 levels about 8 minutes in all. The D=32 levels (about 40 minutes each) are not
# regenerated here.
using RcmpsCore, LinearAlgebra, TOML, SHA, Dates, Printf
BLAS.set_num_threads(1)

const FIXTURES = normpath(joinpath(@__DIR__,"..","test","fixtures"))
const REFERENCES = joinpath(FIXTURES,"excitation_references.toml")
const FOUR = (:standard_energy,:refined_energy,:bilinear_energy,:refined_bilinear_energy)
const USAGE = "usage: julia --project=. --threads=2 benchmark/excitation_references.jl [--update] [--state NAME]...\n" *
              "       julia --project=. benchmark/excitation_references.jl --write-state NAME NEW_FILE.jld2"

function parse_arguments(args)
    update,selected,write = false,String[],nothing
    i = 1
    while i <= length(args)
        if args[i] == "--update"
            update = true
        elseif args[i] == "--state" && i+1 <= length(args)
            push!(selected,args[i+1]);i += 1
        elseif args[i] == "--write-state" && i+2 <= length(args)
            write = (args[i+1],args[i+2]);i += 2
        else
            println(stderr,USAGE);exit(2)
        end
        i += 1
    end
    isnothing(write) || !update && isempty(selected) || (println(stderr,USAGE);exit(2))
    return (;update,selected,write)
end

integrator(t) = IntegratorParameters(n_quad=t["n_quad"],max_point=t["max_point"],atol=t["atol"],
                                     rtol=t["rtol"],maxiters=t["maxiters"])

# maxspace is min(maxspace_cap, n), n the sector dimension.
davidson_options(t,n) = DavidsonOptions(tol=t["tol"],maxops=t["maxops"],maxspace=min(t["maxspace_cap"],n),
    retain=t["retain"],refresh=t["refresh"],hermitian_tol=t["hermitian_tol"],stall_window=t["stall_window"],
    stall_minops=t["stall_minops"],stall_rtol=t["stall_rtol"],stall_residual_limit=t["stall_residual_limit"],
    stall_energy_atol=t["stall_energy_atol"])

function load_state(name,entry)
    path = joinpath(FIXTURES,entry["file"])
    digest = bytes2hex(open(sha256,path))
    digest == entry["sha256"] || error("$(entry["file"]): sha256 $digest, the references record $(entry["sha256"])")
    s = rcmps_load(path)
    s.D == entry["D"] && s.model_param.name == entry["model"] && s.model_param.coupling == entry["coupling"] ||
        error("$(entry["file"]) is not the state [states.$name] describes (D, model or coupling)")
    return s
end

# The commit and the state of src/ of the RcmpsCore that runs, when it is a git checkout.
function rcmpscore_source()
    repo = dirname(dirname(realpath(pathof(RcmpsCore))))
    git(args...) = try
        strip(read(`git --no-optional-locks -C $repo $args`,String))
    catch
        nothing
    end
    head,tree = git("rev-parse","--short=12","HEAD"),git("rev-parse","--short=12","HEAD:src")
    status = git("status","--porcelain","--untracked-files=no","--","src")
    (isnothing(head) || isnothing(tree) || isnothing(status)) && return "unknown (not a git checkout)"
    return "commit $head, src tree $tree" * (isempty(status) ? "" : " with uncommitted changes in src/")
end

function solve(s,settings,sector,momentum,nev)
    s = deepcopy(s)
    s.integrator_param = integrator(settings["integration"])
    julia = settings["julia"]
    julia["ode_method"] == "Vern7" || error("only the library default ODE method Vern7 is supported")
    settings["validate"] || error("reference levels are validated")
    sector,parity,gauge = Symbol(sector),Symbol(settings["parity"]),Symbol(settings["gauge"])
    metric_tol = settings["metric_tol"]
    n = length(TangentCoordinates(s;sector,parity,gauge,metric_tol))
    davidson = davidson_options(settings["davidson"],n)
    model = ExcitationModel(Symbol(s.model_param.name),real(s.model_param.coupling))
    tick = time()
    r = excitation_eigsolve(s,momentum;model,sector,parity,gauge,metric_tol,nev,
        kinetic_nodes=settings["kinetic_nodes"],refined=integrator(settings["refined_integration"]),
        refined_kinetic_nodes=settings["refined_kinetic_nodes"],preconditioner=Symbol(settings["preconditioner"]),
        preconditioner_nodes=settings["preconditioner_nodes"],start=Symbol(settings["start"]),davidson,
        validate=true,validation_tol=settings["validation_tol"],acceptance_residual=settings["acceptance_residual"],
        refine_steps=settings["refine_steps"],environment_tol=julia["environment_tol"],max_bond=julia["max_bond"],
        kinetic_backend=Symbol(julia["kinetic_backend"]),parallel=Symbol(julia["parallel"]))
    return r,time()-tick
end

# The results of a level: --update keeps a level's record when all of them are unchanged.
const RESULTS = ("validated_energy","four_energies","ritz_energy","residual","energy_spread","numops",
                 "action_calls","start","accepted","converged","stop_reason","dimension")

# The fields --update writes for level k of result r.
function computed_fields(r,k,seconds,finished,source)
    v = r.validation[k]
    return Dict{String,Any}(
        "validated_energy"=>r.validated_energies[k],"four_energies"=>[Float64(getfield(v,f)) for f in FOUR],
        "ritz_energy"=>r.energies[k],"residual"=>r.residuals[k],"energy_spread"=>v.energy_spread,
        "numops"=>r.numops,"action_calls"=>r.action_calls,"start"=>string(r.start),"accepted"=>v.accepted,
        "converged"=>r.converged,"stop_reason"=>string(r.stop_reason),"dimension"=>r.dimension,
        "wall_seconds"=>seconds,"computed"=>finished,"julia_version"=>string(VERSION),
        "rcmpscore_version"=>string(pkgversion(RcmpsCore)),"julia_threads"=>Threads.nthreads(),
        "rcmpscore_source"=>source)
end

# One line per compared quantity; returns whether the level passes.
function compare(level,fields)
    name,ok = level["name"],true
    function line(quantity,stored,value,tolerance)
        difference = value-stored
        good = abs(difference) <= tolerance
        ok &= good
        @printf("%-22s %-24s stored %.17g computed %.17g difference % .3e tolerance %.1e %s\n",
                name,quantity,stored,value,difference,tolerance,good ? "ok" : "MISMATCH")
    end
    missing_ = [key for key in ("validated_energy","four_energies","ritz_energy") if !haskey(level,key)]
    if isempty(missing_)
        line("validated_energy",level["validated_energy"],fields["validated_energy"],level["tolerance"])
        for (f,stored,value) in zip(FOUR,level["four_energies"],fields["four_energies"])
            line(string(f),stored,value,level["tolerance"])
        end
        line("ritz_energy",level["ritz_energy"],fields["ritz_energy"],level["ritz_tolerance"])
    else
        println(rpad(name,22)," no stored ",join(missing_,", "));ok = false
    end
    if !(fields["converged"] && fields["accepted"])
        println(rpad(name,22)," NOT ACCEPTED: converged=",fields["converged"]," accepted=",fields["accepted"],
                " stop=",fields["stop_reason"]);ok = false
    end
    changed = [key for key in ("numops","action_calls","start","dimension") if get(level,key,nothing) != fields[key]]
    isempty(changed) || println(rpad(name,22)," note: ",join(("$key $(get(level,key,"-")) -> $(fields[key])"
                                                         for key in changed),", "))
    return ok
end

function write_state(data,name,path)
    ispath(path) && error("$path exists; give a new file")
    entry = data["states"][name];recipe = entry["recipe"]
    recipe["constructor"] == "RandomStateZ2" || error("unsupported recipe constructor $(recipe["constructor"])")
    s = RandomStateZ2(recipe["D"];seed=recipe["seed"],scaling=recipe["scaling"])
    s.model_param = ModelParameters(name=entry["model"],coupling=entry["coupling"])
    s.integrator_param = integrator(recipe["integration"])
    s.info = recipe["info"]
    rcmps_store(s,path;verbose=false)
    println(path," sha256 ",bytes2hex(open(sha256,path)))
end

function main()
    options = parse_arguments(ARGS)
    data = TOML.parsefile(REFERENCES)
    data["format"] == "rcmps-excitation-references" && data["format_version"] == 1 ||
        error("$REFERENCES: unsupported format")
    isnothing(options.write) || return (write_state(data,options.write...);0)
    unknown = setdiff(options.selected,keys(data["states"]))
    isempty(unknown) || error("unknown states: $(join(unknown,", "))")
    levels = [level for level in data["levels"] if level["regenerate"] &&
              (isempty(options.selected) || level["state"] in options.selected)]
    groups = Dict{Any,Vector{Any}}();order = []
    for level in levels
        key = (level["state"],level["settings"],level["sector"],level["momentum"],level["nev"])
        haskey(groups,key) || (groups[key] = [];push!(order,key))
        push!(groups[key],level)
    end
    @printf("%d levels in %d solves; Julia %s, %d threads, BLAS %d thread; RcmpsCore %s, %s\n",length(levels),
            length(order),VERSION,Threads.nthreads(),BLAS.get_num_threads(),pkgversion(RcmpsCore),rcmpscore_source())
    flush(stdout)
    source,states,passed,accepted = rcmpscore_source(),Dict(),true,true
    updates = Pair{Dict{String,Any},Dict{String,Any}}[]
    for key in order
        state,settings,sector,momentum,nev = key
        s = get!(()->load_state(state,data["states"][state]),states,state)
        @printf("solve %s, settings %s, sector %s, p = %s, nev = %d\n",state,settings,sector,repr(momentum),nev)
        flush(stdout)
        r,seconds = solve(s,data["settings"][settings],sector,momentum,nev)
        finished = Dates.format(now(UTC),dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"
        @printf("  %.1f s, numops %d, action calls %d, stop %s, accepted %s\n",seconds,r.numops,r.action_calls,
                r.stop_reason,r.accepted)
        for level in groups[key]
            k = level["level"]
            1 <= k <= nev || error("$(level["name"]): level $k of a solve with nev = $nev")
            fields = computed_fields(r,k,seconds,finished,source)
            passed &= compare(level,fields)
            accepted &= fields["converged"] && fields["accepted"]
            push!(updates,level=>fields)
        end
        flush(stdout)
    end
    if options.update
        accepted || (println("not written: a recomputed level is not converged or not accepted");return 1)
        changed = filter(((level,fields),)->any(key->!isequal(get(level,key,nothing),fields[key]),RESULTS),updates)
        foreach(((level,fields),)->merge!(level,fields),changed)
        temporary = REFERENCES * ".tmp"
        open(io->TOML.print(io,data;sorted=true),temporary,"w")
        mv(temporary,REFERENCES;force=true)
        println("updated $(length(changed)) of $(length(updates)) levels in $REFERENCES (the others are unchanged)")
        return 0
    end
    println(passed ? "all $(length(levels)) levels agree" : "MISMATCH: some levels disagree or are not accepted")
    return passed ? 0 : 1
end

exit(main())
