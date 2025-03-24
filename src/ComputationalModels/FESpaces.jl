
function  Gridap.FESpaces.TestFESpace(model, reffe, bc::DirichletBC ; kwargs...)
     TestFESpace(model, reffe, dirichlet_tags=bc.tags  ; kwargs...)
end

function Gridap.FESpaces.TestFESpace(model, reffe, ::NothingBC ; kwargs...)
     TestFESpace(model, reffe; kwargs...)
end
 
function Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, bc::DirichletBC, Λ::Float64)
    TrialFESpace!(space, map(f -> f(Λ), bc.values))
end

function Gridap.FESpaces.TrialFESpace!(space::SingleFieldFESpace, ::NothingBC, Λ::Float64)
    space
end

function Gridap.FESpaces.TrialFESpace!(space::MultiFieldFESpace, bc::MultiFieldBC, Λ::Float64)
  @inbounds for (i, space) in enumerate(space.spaces)
    TrialFESpace!(space, bc.BoundaryCondition[i], Λ)
  end
end

function Gridap.FESpaces.TrialFESpace(space::SingleFieldFESpace,::NothingBC, Λ::Float64)
  space
end

function Gridap.FESpaces.TrialFESpace(space::SingleFieldFESpace, bc::DirichletBC, Λ::Float64)
    TrialFESpace(space, map(f -> f(Λ), bc.values))
end
  
function Gridap.FESpaces.TrialFESpace(space::MultiFieldFESpace, bc::MultiFieldBC, Λ::Float64)
  U_=Vector{Union{TrialFESpace,UnconstrainedFESpace}}(undef, length(space))
  @inbounds for (i, space) in enumerate(space.spaces)
    U_[i]=TrialFESpace(space, bc.BoundaryCondition[i], Λ)
  end
return  MultiFieldFESpace(U_)

end


# Instantiate nonlinear solver caches (without actually doing the first iteration)

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
  
  function instantiate_caches(x,nls::PETScNonlinearSolver,op::NonlinearOperator)
    return GridapPETSc._setup_cache(x,nls,op)
  end

 






  # function get_FESpaces(::Type{Mechano},
#     model,
#     order::Int64,
#     bconds::DirichletBC; constraint=nothing)

#     # Reference FE
#     reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)

#     # Test FE Spaces
#     V = _TestFESpace(model, reffeu, bconds, :H1)

#     # Trial FE Spaces
#     U =  _TrialFESpace(V,bconds,1.0)

#     return @ntuple V U
# end

# function get_FESpaces!(::Type{Mechano},
#     spaces,
#     bconds::DirichletBC, Λ::Float64)

#     @unpack V = spaces

#     # Trial FE Spaces
#     U =  _TrialFESpace(V,bconds,Λ)

#     spaces = @ntuple V U
# end




# # ========================
# # ThermoMechProblem
# # ========================

# function get_FESpaces(::ThermoMechProblem{:monolithic},
#     model,
#     order::Int64,
#     bconds::MultiFieldBC; constraint=nothing)

#     # Reference FE
#     reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
#     reffeθ = ReferenceFE(lagrangian, Float64, 1)

#     # Test FE Spaces
#     Vu = _TestFESpace(model, reffeu, bconds.BoundaryCondition[1], :H1)
#     Vθ = _TestFESpace(model, reffeθ, bconds.BoundaryCondition[2], :H1)

#     # Trial FE Spaces
#     Uu =  _TrialFESpace(Vu,bconds.BoundaryCondition[1],1.0)
#     Uθ =  _TrialFESpace(Vθ,bconds.BoundaryCondition[2],1.0)

#     # Multifield FE Spaces
#     V = MultiFieldFESpace([Vu, Vθ])
#     U = MultiFieldFESpace([Uu, Uθ])

#     return @ntuple Vu Vθ Uu Uθ V U
# end

# function get_FESpaces!(::ThermoMechProblem{:monolithic},
#     spaces,
#     bconds::MultiFieldBC, Λ::Float64)

#     @unpack Vu, Vθ = spaces

#     # Trial FE Spaces
#     Uu =  _TrialFESpace(Vu,bconds.BoundaryCondition[1],Λ)
#     Uθ =  _TrialFESpace(Vθ,bconds.BoundaryCondition[2],Λ)
    
#     # Multifield FE Spaces
#     V = MultiFieldFESpace([Vu, Vθ])
#     U = MultiFieldFESpace([Uu, Uθ])

#     spaces = @ntuple Vu Vθ Uu Uθ V U
# end


# # ===================
# # ElectroMechProblem
# # ===================

# function get_FESpaces(::ElectroMechProblem{:monolithic},
#     model,
#     order::Int64,
#     bconds::MultiFieldBC; constraint=nothing)


#     # Reference FE
#     reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
#     reffeφ = ReferenceFE(lagrangian, Float64, order)
 
#     # Test FE Spaces
#     Vu = _TestFESpace(model, reffeu, bconds.BoundaryCondition[1], :H1)
#     Vφ = _TestFESpace(model, reffeφ, bconds.BoundaryCondition[2], :H1)

#     # Trial FE Spaces
#     Uu =  _TrialFESpace(Vu,bconds.BoundaryCondition[1],1.0)
#     Uφ =  _TrialFESpace(Vφ,bconds.BoundaryCondition[2],1.0)

#     # Multifield FE Spaces
#     V = MultiFieldFESpace([Vu, Vφ])
#     U = MultiFieldFESpace([Uu, Uφ])

#     return @ntuple Vu Vφ Uu Uφ V U
# end

# function get_FESpaces!(::ElectroMechProblem{:monolithic},
#     spaces,
#     bconds::MultiFieldBC, Λ::Float64)

#     @unpack Vu, Vφ = spaces

#     # Trial FE Spaces
#     Uu =  _TrialFESpace(Vu,bconds.BoundaryCondition[1],Λ)
#     Uφ =  _TrialFESpace(Vφ,bconds.BoundaryCondition[2],Λ)

#     # Multifield FE Spaces
#     V = MultiFieldFESpace([Vu, Vφ])
#     U = MultiFieldFESpace([Uu, Uφ])

#     spaces = @ntuple Vu Vφ Uu Uφ V U
# end

# # ========================
# # ThermoElectroMechProblem
# # ========================

# function get_FESpaces(::ThermoElectroMechProblem{:monolithic},
#     model,
#     order::Int64,
#     bconds::MultiFieldBC; constraint=nothing)

#     # Reference FE
#     reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
#     reffeφ = ReferenceFE(lagrangian, Float64, order)
#     reffeθ = ReferenceFE(lagrangian, Float64, 1)

#     # Test FE Spaces
#     Vu = _TestFESpace(model, reffeu, bconds.BoundaryCondition[1], :H1)
#     Vφ = _TestFESpace(model, reffeφ, bconds.BoundaryCondition[2], :H1)
#     Vθ = _TestFESpace(model, reffeθ, bconds.BoundaryCondition[3], :H1)

#     # Trial FE Spaces
#     Uu =  _TrialFESpace(Vu,bconds.BoundaryCondition[1],1.0)
#     Uφ =  _TrialFESpace(Vφ,bconds.BoundaryCondition[2],1.0)
#     Uθ =  _TrialFESpace(Vθ,bconds.BoundaryCondition[3],1.0)

#     # Multifield FE Spaces
#     V = MultiFieldFESpace([Vu, Vφ, Vθ])
#     U = MultiFieldFESpace([Uu, Uφ, Uθ])

#     return @ntuple Vu Vφ Vθ Uu Uφ Uθ V U
# end

# function get_FESpaces!(::ThermoElectroMechProblem{:monolithic},
#     spaces,
#     bconds::MultiFieldBC, Λ::Float64)

#     @unpack Vu, Vφ, Vθ = spaces

#     # Trial FE Spaces
#     Uu =  _TrialFESpace(Vu,bconds.BoundaryCondition[1],Λ)
#     Uφ =  _TrialFESpace(Vφ,bconds.BoundaryCondition[2],Λ)
#     Uθ =  _TrialFESpace(Vθ,bconds.BoundaryCondition[3],Λ)
    
#     # Multifield FE Spaces
#     V = MultiFieldFESpace([Vu, Vφ, Vθ])
#     U = MultiFieldFESpace([Uu, Uφ, Uθ])

#     spaces = @ntuple Vu Vφ Vθ Uu Uφ Uθ V U
# end






