# The missing boundary gradient

This derivation uses the left-canonical, single-field conventions of this repository,
with bare mass one. The independent implementation in `benchmark/boundary_reference.py`
was written and run before the Julia observable engine. It uses SciPy DOP853,
adaptive quadrature, and dense linear algebra rather than the Julia solvers.

## Finite-window differentiation

Let

\[
\mathcal L(X)=QX+XQ^\dagger+RXR^\dagger,
\qquad Q=-iK-\tfrac12R^\dagger R,
\qquad \mathcal L(\rho_0)=0,\quad\operatorname{tr}\rho_0=1.
\]

For a vertex operator, the actual numerical objective is

\[
F_L(Q,R)=\operatorname{tr}\big[U(L,-L)\rho_0(Q,R)\big],\qquad
\partial_x U=(\mathcal L+\beta J(x)\mathcal R)U,
\quad \mathcal R(X)=RX+XR^\dagger.
\]

The derivative has **two** terms:

\[
\delta F_L=\int_{-L}^L\operatorname{tr}\big[O_x\,\delta\mathcal L_\beta(x)(\rho_x)\big]\,dx
+\operatorname{tr}\big[O_{-L}\,\delta\rho_0\big].
\]

Here \(O_x\) is the trace-dual backward solution, with \(O_L=I\).
For real beta it is Hermitian. For complex beta the trace-dual equation retains
beta rather than conjugating it. There is no corresponding right-boundary derivative in this gauge: the terminal
identity is parameter-independent, and source-free propagation preserves trace.
The infinite-line adjoint formula already includes the initial-state response as
its remote left tail. The error arises when that integral is truncated while
rho0 is still recomputed as a function of Q and R.

The first term is the bulk gradient in the old
implementation. Its initial state is recomputed when moving Q and R, but that
state's derivative was absent from the gradient calculation.

Differentiating stationarity gives

\[
\mathcal L(\delta\rho_0)=-\delta\mathcal L(\rho_0),\qquad
\operatorname{tr}\delta\rho_0=0.
\]

Set \(A=\operatorname{Herm}(O_{-L})\) when differentiating the real part of the
observable. The Hermitian solution H of

\[
\boxed{\big[-\mathcal L^*+I\operatorname{tr}(\rho_0\,\cdot)\big]H
=A-\operatorname{tr}(\rho_0 A)I}
\]

satisfies \(\operatorname{tr}(\rho_0H)=0\). Assuming a unique stationary state,
the pinned operator is invertible. Consequently

\[
\boxed{\operatorname{Re}\operatorname{tr}(O_{-L}\delta\rho_0)
=\operatorname{tr}[H\,\delta\mathcal L(\rho_0)].}
\]

Thus one Poisson solve gives all parameter derivatives at once. There is no
finite difference of rho in production and no evolution over a time 1/gap.

## Horizontal tangent and metric conventions

The canonical retraction has tangent \(\delta R=W\), \(\delta Q=-R^\dagger W\).
It follows that

\[
\delta\mathcal L(\rho_0)=-R^\dagger W\rho_0-\rho_0W^\dagger R
+W\rho_0R^\dagger+R\rho_0W^\dagger.
\]

Writing a real derivative as \(2\operatorname{Re}\operatorname{tr}(MW)\), its
boundary covector is

\[
\boxed{M_{\rm boundary}=\rho_0R^\dagger H-\rho_0HR^\dagger.}
\]

With the existing regularized metric
\(g(W_1,W_2)=2\operatorname{Re}\operatorname{tr}(W_1(\rho_0+\epsilon I)W_2^\dagger)\),
the returned gradient is \(G=[(\rho_0+\epsilon I)^{-1}M]^\dagger\).
In particular, `metric(s,G,W)` is the derivative; there is no extra factor two
to apply at call sites.

For a complex vertex, separately accumulate the coefficients M and N of W and
\(W^\dagger\), then replace the real-objective covector by
\((M+N^\dagger)/2\). Assuming these two coefficients are conjugates would be wrong
for imaginary beta. `vertex` returns the complex value and the gradient of its
real part. This suffices for the real sine-Gordon energy.

## Why a small gap matters

Outside the source region, the identity component of the backward environment
is irrelevant because \(\operatorname{tr}\delta\rho_0=0\). Its remaining part
relaxes under \(\mathcal L^*\), with an asymptotic rate set by the transfer gap
\(\Delta\), not the mass in J. Ignoring the initial-state term therefore leaves a
contribution behaving roughly as \(e^{-\Delta L}\), with a susceptibility
prefactor from the inverse Lindbladian. Nonnormality, Jordan blocks, and
cancellations prevent a universal monotone pure exponential bound.

The correction differentiates the **finite-window objective exactly**, up to ODE,
quadrature, and linear-solver errors. Truncating J still changes the infinite-line
observable: J has the bare-mass exponential tail (times an algebraic factor),
and its derivative can have gap-dependent prefactors. The correction removes
the separate transfer-tail omission; it does not eliminate source truncation,
conditioning, or a physical divergence at a nonunique stationary point.

## Other observables

Field monomials are an augmented triangular system with initial channel
\(\rho^{(0)}=\rho_0\) and all other channels zero. Both kinetic terms use the four
channels \((\rho_0,\rho_1,\rho_2,\rho_{12})\), with insertions R or \([Q,R]\).
The same boundary formula applies to the backward environment of channel zero.
Keeping that channel explicit accounts for the appearances of rho0 throughout
the kinetic equations. The commutator insertion is differentiated through both
Q and R. For an energy sum, covectors and boundary right-hand sides are summed
before one shared Poisson solve and one metric solve.

## Implementation of the Poisson solve

The production method applies the pinned operator without forming a D² by D²
matrix, and uses right-preconditioned GMRES. A Schur factorization of Q provides
the inverse of the no-jump operator, with the rank-one pin included by
Sherman–Morrison. True residuals are checked after solving. A stalled iterative solve at D≤16
falls back to a dense solve with the same residual criterion. A dense method is
available as an independent reference and is used in tests.

The steady-state solver uses the CPTP map
\(\Phi=\mathcal K(-\mathcal S)^{-1}\) from Beugnot, Gregory, Robin and Tilloy,
then reconstructs \(\rho_0\propto(-\mathcal S)^{-1}\xi\). The order of these
operators matters: iterating \((-\mathcal S)^{-1}\mathcal K\) is a different map.
Largest-real-part Arnoldi targets +1, including cases with other peripheral
phases. For singular no-jump evolution, a dummy identity jump and compensating
no-jump shift leave the physical Lindbladian unchanged while making this split
invertible. This avoids a special pure-state eigenvector heuristic.

Each Schur inverse application and Lindblad action costs O(D³); a Krylov solve's
iteration count remains problem-dependent. Memory is O(kD²) for Krylov dimension
k. At D≤8 a dense singular-value check additionally rejects a nonunique or
numerically unresolved stationary state. At larger D, uniqueness is an explicit
assumption; a small residual alone does not certify it or bound forward error
without conditioning information. The `:dense` reference also checks uniqueness.

## Sources

- Antoine Tilloy, [Relativistic continuous matrix product states for quantum fields without cutoff](https://arxiv.org/abs/2102.07741), especially the observable ODE and adjoint construction.
- Gaspard Beugnot, Paul Gregory, Rémi Robin, Antoine Tilloy, [Well-conditioned iterative methods for large open quantum systems](https://arxiv.org/abs/2608.30860), sections 2, 3 and appendix A.

The finite-window correction and its benchmarking here are derived from the
stationarity equation above, rather than quoted as a result of either paper.
