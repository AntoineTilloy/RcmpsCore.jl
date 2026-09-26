module RcmpsCore
using LinearAlgebra, Random, KrylovKit, SpecialFunctions
using SciMLBase: ODEProblem, solve, successful_retcode
using OrdinaryDiffEqVerner: Vern7
import OptimKit
import JLD2

export State, RandomState, RandomStateZ2, IntegratorParameters, ModelParameters, ManifoldParameters
export lindblad, lindblad_adjoint, lindblad!, lindblad_adjoint!, generate_T, mass_gap
export steady_state, get_ρ0, boundary_environment, SolverFailure, entanglement_entropy, robust_inv
export jsource, vertex, phi_monomial, kinetic1, kinetic2, fg_phi4, fg_sgordon, fg_phi4_massmod
export metric, precondition, retraction, retraction_regulated, naive_transport!, finalize!, copy_param!
export optimize_state, SafeguardedLineSearch, ls_default, ls_lenient, ls_backward
export KineticOptions, KineticKrylovOptions
export transfer_gap, ExcitationModel, tangent_metric, excitation_environments
export excitation_matrix_element, excitation_hamiltonian, excitation_spectrum, excitation_gap
export excitation_hamiltonian_action, excitation_eigsolve, DavidsonOptions, TangentCoordinates
export ExcitationContext, excitation_parity
export dispersion_relation, relativistic_diagnostics
export rcmps_store, rcmps_load, get_states, get_exact_state, select_best

include("state.jl")
include("transfer.jl")
include("kinetic.jl")
include("kinetic_krylov.jl")
include("observables.jl")
include("excitation_environments.jl")
include("excitations.jl")
include("excitation_action.jl")
include("davidson.jl")
include("excitation_eigsolve.jl")
include("optimization.jl")
include("storage.jl")
end
