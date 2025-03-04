
"""
    struct IterativeSolver <: LinearSolver end

Wrapper of the  solvers available in IterativeSolvers package in julia
"""
#constructor
struct IterativeSolver <: Gridap.Algebra.LinearSolver
  solver::String
  kwargs::Dict
end

function IterativeSolver(solver::String; kwargs...)
  IterativeSolver(solver, kwargs)
end


# SymbolicSetup
struct IterativeSolverss <: Gridap.Algebra.SymbolicSetup
  itsolver::IterativeSolver
end


function Gridap.Algebra.symbolic_setup(solver::IterativeSolver, mat::AbstractMatrix)
  IterativeSolverss(solver)
end

# NumericalSetup
mutable struct IterativeSolverns{T<:AbstractMatrix} <: Gridap.Algebra.NumericalSetup
  A::T
  itsolver::IterativeSolver
  kwargs::Dict
end

function Gridap.Algebra.numerical_setup(ss::IterativeSolverss, mat::AbstractMatrix)
  kwargs_ = copy(ss.itsolver.kwargs)
  if haskey(kwargs_, :Pl)
    kwargs_[:Pl] = kwargs_[:Pl](mat)
  end
  IterativeSolverns(mat, ss.itsolver, kwargs_)
end

function Gridap.Algebra.numerical_setup!(ns::IterativeSolverns, mat::AbstractMatrix)
  ns.A = mat
  if haskey(ns.kwargs, :Pl)
    ns.kwargs[:Pl] = ns.itsolver.kwargs[:Pl](mat)
  end
  ns
end


function Gridap.Algebra.solve!(
  x::AbstractVector, ns::IterativeSolverns, b::AbstractVector)
  fexp = eval(Meta.parse(ns.itsolver.solver))
  copy_entries!(x, fexp(ns.A, b; ns.kwargs...))
end

 
