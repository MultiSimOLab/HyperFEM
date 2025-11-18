
function Gridap.FESpaces.TestFESpace(model, reffe, bc::DirichletBC; kwargs...)
  TestFESpace(model, reffe, dirichlet_tags=bc.tags; kwargs...)
end

function Gridap.FESpaces.TestFESpace(model, reffe, ::NothingBC; kwargs...)
  TestFESpace(model, reffe; kwargs...)
end

function Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, bc::DirichletBC, Λ::Float64)
  #   TrialFESpace!(space, map(f -> f(Λ), bc.values)) 
  values = Vector{Any}(undef, length(bc.tags))
  @inbounds for i in eachindex(bc.tags)
    if bc.caches[i] isa Union{Vector,Float64}
      values[i] = bc.values[i](Λ)(0.0)
    else
      values[i] = bc.values[i](Λ)
    end
  end
  TrialFESpace!(space, values)
  ## actualizar DirichletCoupling
  @inbounds for i in eachindex(bc.tags)
    if bc.caches[i] isa InterpolableBC
      bc.caches[i](Λ)
      space.dirichlet_values[bc.caches[i].caches[2]] = bc.caches[i].caches[1]
    end
  end
  return space
end


function Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, bc::DirichletBC, Λ::Float64, ΔΛ::Float64)
  # TrialFESpace!(space, map(f -> ((x) -> f(Λ)(x) - f(Λ - ΔΛ)(x)), bc.values))
  values = Vector{Any}(undef, length(bc.tags))
  @inbounds for i in eachindex(bc.tags)
    if bc.caches[i] isa Union{Vector,Float64}
      values[i] = bc.values[i](Λ)(0.0) - bc.values[i](Λ - ΔΛ)(0.0)
    else
      values[i] = (x) -> bc.values[i](Λ)(x) - bc.values[i](Λ - ΔΛ)(x)
    end
  end
  TrialFESpace!(space, values)

  @inbounds for i in eachindex(bc.tags)
    if bc.caches[i] isa InterpolableBC
      bc.caches[i](Λ - ΔΛ)
      v0 = copy(bc.caches[i].caches[1])
      bc.caches[i](Λ)
      space.dirichlet_values[bc.caches[i].caches[2]] = bc.caches[i].caches[1] - v0
    end
  end
  return space
end


function Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, ::NothingBC, Λ::Float64)
  space
end

function Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, ::NothingBC, Λ::Float64, ΔΛ::Float64)
  space
end

function Gridap.FESpaces.TrialFESpace!(space::MultiFieldFESpace, bc::MultiFieldBC, Λ::Float64)
  @inbounds for (i, space) in enumerate(space.spaces)
    TrialFESpace!(space, bc[i], Λ)
  end
end

function Gridap.FESpaces.TrialFESpace!(space::MultiFieldFESpace, bc::MultiFieldBC, Λ::Float64, ΔΛ::Float64)
  @inbounds for (i, space) in enumerate(space.spaces)
    TrialFESpace!(space, bc[i], Λ, ΔΛ)
  end
end


function Gridap.FESpaces.TrialFESpace(space::SingleFieldFESpace, ::NothingBC, Λ::Float64=0.0)
  space
end


function Gridap.FESpaces.TrialFESpace(space::SingleFieldFESpace, bc::DirichletBC, Λ::Float64=0.0)
  #     trialspace= TrialFESpace(space, map(f -> f(Λ), bc.values))

  values = Vector{Any}(undef, length(bc.tags))

  @inbounds for i in eachindex(bc.tags)
    if bc.caches[i] isa Union{Vector,Float64}
      values[i] = bc.values[i](Λ)(0.0)
    else
      values[i] = bc.values[i](Λ)
    end
  end

  trialspace = TrialFESpace(space, values)
  @inbounds for i in eachindex(bc.tags)
    if bc.caches[i] isa InterpolableBC
      bc.caches[i](Λ)
      trialspace.dirichlet_values[bc.caches[i].caches[2]] = bc.caches[i].caches[1]
    end
  end
  return trialspace
end

function Gridap.FESpaces.TrialFESpace(space::MultiFieldFESpace, bc::MultiFieldBC, Λ::Float64=0.0)
  U_ = Vector{Union{TrialFESpace,UnconstrainedFESpace}}(undef, length(space))
  @inbounds for (i, space) in enumerate(space.spaces)
    U_[i] = TrialFESpace(space, bc[i], Λ)
  end
  return MultiFieldFESpace(U_)
end



function instantiate_caches(x, nls::NLSolver, op::NonlinearOperator)
  Gridap.Algebra._new_nlsolve_cache(x, nls, op)
end

function instantiate_caches(x, nls::NewtonRaphsonSolver, op::NonlinearOperator)
  b = residual(op, x)
  A = jacobian(op, x)
  dx = similar(b)
  ss = symbolic_setup(nls.ls, A)
  ns = numerical_setup(ss, A)
  return Gridap.Algebra.NewtonRaphsonCache(A, b, dx, ns)
end

function instantiate_caches(x, nls::NewtonSolver, op::NonlinearOperator)
  b = residual(op, x)
  A = jacobian(op, x)
  dx = allocate_in_domain(A)
  fill!(dx, zero(eltype(dx)))
  ss = symbolic_setup(nls.ls, A)
  ns = numerical_setup(ss, A, x)
  return GridapSolvers.NonlinearSolvers.NewtonCache(A, b, dx, ns)
end

function instantiate_caches(x, nls::Newton_RaphsonSolver, op::NonlinearOperator)
  b = residual(op, x)
  A = jacobian(op, x)
  dx = allocate_in_domain(A)
  fill!(dx, zero(eltype(dx)))
  ss = symbolic_setup(nls.ls, A)
  ns = numerical_setup(ss, A, x)
  return Newton_RaphsonCache(A, b, dx, ns)
end


function instantiate_caches(x, nls::PETScNonlinearSolver, op::NonlinearOperator)
  return GridapPETSc._setup_cache(x, nls, op)
end






