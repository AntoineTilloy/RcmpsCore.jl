# The trial steps of OptimKit's HagerZhangLineSearch as RcmpsCore's SafeguardedLineSearch
# builds it (HagerZhangLineSearch(ϵ=1e-10), with acceptfirst=true as OptimKit's LBFGS passes)
# on one-dimensional functions phi(alpha) with the slope phi'(alpha): the golden values of
# gpu/tests/optimizer/test_line_search.py, which runs the port of the GPU optimizer's robust
# controller (runtime/robust_optimizer.py hager_zhang) on the same functions. Powers are written
# as products so that Python evaluates the same floating-point expressions.
#
# Usage, from the repository root (any Julia environment with OptimKit 0.5.1):
#   julia --project=. gpu/qualification/optimizer/hager_zhang_references.jl
#
# Prints one line per function: name, the step the search returns, every trial step in order,
# and the number of evaluations (OptimKit 0.5.1, on 2026-10-26: the GOLDEN table of the test).
using OptimKit, Printf
wall(x, l, u, h) = l < x < u ? h : 0.0
cases = [
    ("quadratic_min_3",  a -> ((a-3)*(a-3), 2(a-3)), 1.0),
    ("quadratic_min_20", a -> ((a-20)*(a-20), 2(a-20)), 1.0),
    ("exponential",      a -> (exp(a)-5a, exp(a)-5), 10.0),
    ("offset_energy",    a -> (1e-12*(a-1)*(a-1) + (a == 0 ? 0.0 : 5e-12), 2e-12*(a-1)), 3.0),
    ("energy_wall",      a -> (-a + (a > 0.5 ? 1e3 : 0.0), -1.0), 1.0),
    ("cosine",           a -> (cos(0.3+a) + 0.1*(0.3+a), -sin(0.3+a) + 0.1), 1.0),
    ("linear",           a -> (-a, -1.0), 1e-3),
    ("steep_wall",       a -> (-a + 50*max(0.0, a-0.7)*max(0.0, a-0.7), -1 + 100*max(0.0, a-0.7)), 8.0),
    ("gaussian_bump",    a -> (-3a + a*a*a/3 + 3exp(-((a-0.3)/0.15)*((a-0.3)/0.15)),
                               -3 + a*a - 2*3*(a-0.3)/(0.15*0.15)*exp(-((a-0.3)/0.15)*((a-0.3)/0.15))), 4.0),
    ("cubic_wall",       a -> (-1.5a + a*a*a/3 + wall(a, 0.2, 0.5, 3.0), -1.5 + a*a), 4.0),
]
ls = HagerZhangLineSearch(ϵ=1e-10)
println("# OptimKit ", pkgversion(OptimKit), "; c1=", ls.c₁, " c2=", ls.c₂, " eps=", ls.ϵ, " theta=", ls.θ,
        " gamma=", ls.γ, " rho=", ls.ρ, " maxiter=", ls.maxiter, " maxfg=", ls.maxfg)
for (name, f, alpha0) in cases
    steps = Float64[]
    fg = a -> (push!(steps, a); f(a))
    _, _, _, _, alpha, nfg = ls(fg, 0.0, 1.0, f(0.0); initialguess=alpha0, acceptfirst=true, verbosity=0)
    @printf("%s %s [%s] %d\n", name, repr(alpha), join(repr.(steps), ", "), nfg)
end
