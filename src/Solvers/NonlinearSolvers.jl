
"""
    struct Newton_RaphsonSolver <: Algebra.NonlinearSolver
  
  Newton-Raphson solver. Same as `NewtonSolver` in GridapSolvers,
"""
struct Newton_RaphsonSolver <: Algebra.NonlinearSolver
  ls::Algebra.LinearSolver
  log::ConvergenceLog{Float64}
end

function Newton_RaphsonSolver(ls; maxiter=100, atol=1e-12, rtol=1.e-6, verbose=0, name="Newton-Raphson")
  tols = SolverTolerances{Float64}(; maxiter=maxiter, atol=atol, rtol=rtol)
  log = ConvergenceLog(name, tols; verbose=verbose)
  return Newton_RaphsonSolver(ls, log)
end

AbstractTrees.children(s::Newton_RaphsonSolver) = [s.ls]

struct Newton_RaphsonCache
  A::AbstractMatrix
  b::AbstractVector
  dx::AbstractVector
  ns::NumericalSetup
end

function Algebra.solve!(x::AbstractVector, nls::Newton_RaphsonSolver, op::NonlinearOperator, cache::Nothing)
  b = residual(op, x)
  A = jacobian(op, x)
  dx = allocate_in_domain(A)
  fill!(dx, zero(eltype(dx)))
  ss = symbolic_setup(nls.ls, A)
  ns = numerical_setup(ss, A, x)
  _solve_nr!(x, A, b, dx, ns, nls, op)
  return Newton_RaphsonCache(A, b, dx, ns)
end

function Algebra.solve!(x::AbstractVector, nls::Newton_RaphsonSolver, op::NonlinearOperator, cache::Newton_RaphsonCache)
  A, b, dx, ns = cache.A, cache.b, cache.dx, cache.ns
  residual!(b, op, x)
  jacobian!(A, op, x)
  numerical_setup!(ns, A, x)
  _solve_nr!(x, A, b, dx, ns, nls, op)
  return cache
end

function _solve_nr!(x, A, b, dx, ns, nls, op)
  log = nls.log

  # Check for convergence on the initial residual
  res = norm(b)
  done = init!(log, res)

  # Newton-like iterations
  while !done

    # Solve linearized problem
    rmul!(b, -1)
    solve!(dx, ns, b)

    #linesearch x and b changed
    # if abs(b' * dx) < 1e-6
    #   break
    # end
      # x .+= dx

      _RPlinesearch!(x, dx, b, op)

    # Check convergence for the current residual
    residual!(b, op, x)
    res = norm(b)
    done = update!(log, res)

    if !done
      # Update jacobian and solver
      jacobian!(A, op, x)
      numerical_setup!(ns, A, x)
    end

  end

  finalize!(log, res)
  return x
end

function _RPlinesearch!(x::AbstractVector, dx::AbstractVector, b::AbstractVector, op::NonlinearOperator; maxiter=50, αmin=1e-16, ρ=0.5, c=0.95)

  m = 0
  α = 1.0
  R₀ = b' * dx

  while α > αmin && m < maxiter
    residual!(b, op, x + α * dx)
    R = b' * dx
    if R <= c * R₀
      break
    end
    α *= ρ
    m += 1
  end
  @show α

  x .+= α * dx
end