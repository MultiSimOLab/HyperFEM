
function  Gridap.FESpaces.TestFESpace(model, reffe, bc::DirichletBC ; kwargs...)
     TestFESpace(model, reffe, dirichlet_tags=bc.tags  ; kwargs...)
end

function Gridap.FESpaces.TestFESpace(model, reffe, ::NothingBC ; kwargs...)
     TestFESpace(model, reffe; kwargs...)
end
 
function Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, bc::DirichletBC, Λ::Float64)
    TrialFESpace!(space, map(f -> f(Λ), bc.values))
        ## actualizar DirichletCoupling

end

 
function Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, bc::DirichletBC, Λ::Float64, ΔΛ::Float64)
  TrialFESpace!(space, map(f -> ((x)->f(Λ)(x)-f(Λ-ΔΛ)(x)), bc.values))
      ## actualizar DirichletCoupling

end


function Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, ::NothingBC, Λ::Float64)
    space
end

function Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, ::NothingBC, Λ::Float64, ΔΛ::Float64)
  space
end

function Gridap.FESpaces.TrialFESpace!(space::MultiFieldFESpace, bc::MultiFieldBC, Λ::Float64)
  @inbounds for (i, space) in enumerate(space.spaces)
    TrialFESpace!(space, bc.BoundaryCondition[i], Λ)
  end
end

function Gridap.FESpaces.TrialFESpace!(space::MultiFieldFESpace, bc::MultiFieldBC, Λ::Float64, ΔΛ::Float64)
  @inbounds for (i, space) in enumerate(space.spaces)
    TrialFESpace!(space, bc.BoundaryCondition[i], Λ, ΔΛ)
  end
end


function Gridap.FESpaces.TrialFESpace(space::SingleFieldFESpace,::NothingBC, Λ::Float64)
  space
end

function Gridap.FESpaces.TrialFESpace(space::SingleFieldFESpace, bc::DirichletBC, Λ::Float64)
    TrialFESpace(space, map(f -> f(Λ), bc.values))
    ## actualizar DirichletCoupling
    ## idea si bc.timesteps[i] isa DirichletCoupling entonces meto valor zero y a posteriori machaco
end
  
function Gridap.FESpaces.TrialFESpace(space::MultiFieldFESpace, bc::MultiFieldBC, Λ::Float64)
  U_=Vector{Union{TrialFESpace,UnconstrainedFESpace}}(undef, length(space))
  @inbounds for (i, space) in enumerate(space.spaces)
    U_[i]=TrialFESpace(space, bc.BoundaryCondition[i], Λ)
  end
return  MultiFieldFESpace(U_)

end



function instantiate_caches(x,nls::NLSolver,op::NonlinearOperator)
    Gridap.Algebra._new_nlsolve_cache(x,nls,op)
  end
  
  function instantiate_caches(x,nls::NewtonRaphsonSolver,op::NonlinearOperator)
    b = residual(op, x)
    A = jacobian(op, x)
    dx = similar(b)
    ss = symbolic_setup(nls.ls, A)
    ns = numerical_setup(ss,A)
    return Gridap.Algebra.NewtonRaphsonCache(A,b,dx,ns)
  end
  
  function instantiate_caches(x,nls::NewtonSolver,op::NonlinearOperator)
    b  = residual(op, x)
    A  = jacobian(op, x)
    dx = allocate_in_domain(A); fill!(dx,zero(eltype(dx)))
    ss = symbolic_setup(nls.ls, A)
    ns = numerical_setup(ss,A,x)
    return GridapSolvers.NonlinearSolvers.NewtonCache(A,b,dx,ns)
  end
  
  function instantiate_caches(x,nls::Newton_RaphsonSolver,op::NonlinearOperator)
    b  = residual(op, x)
    A  = jacobian(op, x)
    dx = allocate_in_domain(A); fill!(dx,zero(eltype(dx)))
    ss = symbolic_setup(nls.ls, A)
    ns = numerical_setup(ss,A,x)
    return Newton_RaphsonCache(A,b,dx,ns)
  end
  

  function instantiate_caches(x,nls::PETScNonlinearSolver,op::NonlinearOperator)
    return GridapPETSc._setup_cache(x,nls,op)
  end

 




