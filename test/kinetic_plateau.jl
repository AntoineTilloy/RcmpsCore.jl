# Controlled sequences exercise the acceptance policy independently of roundoff.
struct KineticPlateauFixture
    values::Dict{Int,Tuple{Float64,Float64}}
    visited::Vector{Int}
end
RcmpsCore.kinetic_context(s,::Val{:plateau_test},options)=nothing
function RcmpsCore.kinetic_sum(s,terms::KineticPlateauFixture,::Nothing,n;gradient)
    push!(terms.visited,n)
    e,g=terms.values[n]
    return gradient ? (e,fill(complex(g*(1+s.manifold_param.metric_reg)),1,1),zeros(ComplexF64,1,1)) : e
end
@testset "Checked kinetic gradient plateau" begin
    s=State(reshape(ComplexF64[.3],1,1),zeros(1,1))
    function evaluate(values;options=KineticOptions(maxnodes=128,gradient_fallback_atol=1e-8),gradient=true)
        fixture=KineticPlateauFixture(Dict(n=>v for (n,v) in zip((32,64,128),values)),Int[])
        result=RcmpsCore.kinetic_resolvent(s,fixture;backend=:plateau_test,options,gradient)
        result,fixture.visited
    end
    plateau=[(0.,0.),(0.,1e-10),(0.,0.)]
    result,visited=@test_logs (:warn,r"accepted gradient plateau") evaluate(plateau)
    @test visited==[32,64,128] # two refinements, not one
    @test result[1]==0
    @test_throws SolverFailure evaluate(plateau;options=KineticOptions(maxnodes=128))
    @test_throws SolverFailure evaluate([(0.,0.),(0.,1e-3),(0.,0.)])
    @test_throws SolverFailure evaluate([(0.,0.),(1e-4,1e-10),(1e-4,0.)]) # previous energy check failed
    @test_throws SolverFailure evaluate([(0.,0.),(0.,1e-10),(1e-4,0.)]) # current energy check failed
    @test_throws SolverFailure evaluate([(0.,0.),(0.,1e-9),(0.,1.25e-9)]) # still decreasing rapidly
    @test_throws SolverFailure evaluate([(0.,0.),(0.,Inf),(0.,0.)])
    result,visited=evaluate([(0.,0.),(0.,1e-14),(0.,0.)])
    @test visited==[32,64] # strict path still returns promptly
    result,visited=evaluate(plateau;gradient=false)
    @test result==0 && visited==[32,64]
    for tolerance in (-1.,Inf,NaN)
        @test_throws ArgumentError evaluate(plateau;options=KineticOptions(gradient_fallback_atol=tolerance))
    end
    for ratio in (0.,-1.,1.1,Inf,NaN)
        @test_throws ArgumentError evaluate(plateau;options=KineticOptions(gradient_plateau_ratio=ratio))
    end
end
