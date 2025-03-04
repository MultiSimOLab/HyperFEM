include("../src/Mimosa.jl")

using Gridap.TensorValues
using Gridap
using DrWatson
using GridapGmsh
 
function ϝ(v::Float64)
    (x) -> v
end

function ϝ(v::Vector{Float64})
    (x) -> VectorValue(v)
end



abstract type BoundaryCondition end
struct NothingBC<: BoundaryCondition end


struct MultiFieldBoundaryCondition <: BoundaryCondition
    BoundaryCondition::Vector{BoundaryCondition}
end


function _get_bc_func(tags_::Vector{String}, values_,  bc_timesteps)
    bc_func_ = Vector{Function}(undef, length(tags_))
    @inbounds for i in eachindex(tags_)
        @assert(length(tags_) == length(values_))
        # get funcion generators for boundary conditions
        u_bc(Λ::Float64) = (x) -> ϝ(values_[i])(x) * bc_timesteps[i](Λ)
        bc_func_[i] = u_bc
    end
    return (bc_tags=tags_, bc_func=bc_func_,)
end
 


struct NeumannBC <: BoundaryCondition
    tags::Vector{String}         # tags for boundary conditions
    values::Vector{Function}     # f(x)
    timesteps::Vector{Function}  # f(Λ)

    function NeumannBC(bc_tags::Vector{String}, bc_values, bc_timesteps)  
        @assert(length(bc_tags) == length(bc_values) == length(bc_timesteps))
        tags_,funcs_=_get_bc_func(bc_tags, bc_values, bc_timesteps)
        new(tags_, funcs_, bc_timesteps)
    end
end

evolφ(Λ) = Λ
neu_φ_tags = ["topsuf"]
neu_φ_values = [0.3]
neu_φ_timesteps = [evolφ]
Nφ = NeumannBC(neu_φ_tags, neu_φ_values, neu_φ_timesteps)

neumannbc = MultiFieldBoundaryCondition([NothingBC(), Nφ])




struct CouplingStrategy{Kind} end
 

model = GmshDiscreteModel(datadir("models", "ex2_mesh.msh"))

degree = 2 

 


#------------------------------------------------------------
#                   Neumann Boundary conditions measures
#------------------------------------------------------------


function get_Neumann_dΓ(model,bc::NeumannBC,degree)
    dΓ=Vector{Gridap.CellData.GenericMeasure}(undef, length(bc.tags))
    for i in 1:length(bc.tags)
        Γ= BoundaryTriangulation(model, tags=bc.tags[i])
        dΓ[i]= Measure(Γ, degree)
    end
    return dΓ
end

function get_Neumann_dΓ(model,::NothingBC,degree)
    Vector{Gridap.CellData.GenericMeasure}(undef, 1)
end
 
function get_Neumann_dΓ(model,bc::MultiFieldBoundaryCondition,degree::Int64)
    dΓ=Vector{Vector{Gridap.CellData.GenericMeasure}}(undef, length(bc.BoundaryCondition))
    for (i,bc_i) in enumerate(bc.BoundaryCondition)
        dΓ[i]= get_Neumann_dΓ(model,bc_i,degree)
    end
    return dΓ
end




#------------------------------------------------------------
#                   Neumann Boundary conditions residuals
#------------------------------------------------------------

function residual_Neumann(::NothingBC, v, dΓ;  Λ=1.0) end

function residual_Neumann(bc::NeumannBC, v, dΓ;  Λ=1.0)
    bc_func_ = Vector{Function}(undef, length(bc.tags))
     for (i,f) in enumerate(bc.values)
        bc_func_[i]=(v)->∫(v⋅f(Λ))dΓ[i]
     end
     return mapreduce(f -> f(v), +, bc_func_)
end

function residual_Neumann_EM(::Mimosa.WeakForms.CouplingStrategy{:monolithic}, (v, vφ), (bc,bcφ), (dΓ,dΓφ);  Λ=1.0)
     residual_Neumann(bc, v, dΓ;  Λ=Λ)+
     residual_Neumann(bcφ, vφ, dΓφ;  Λ=Λ)
end
 
 



dΓ=get_Neumann_dΓ(model,neumannbc, degree)


 

 
import Base: +
function (+)(::Nothing, b::Gridap.CellData.DomainContribution)
b
end
function (+)(b::Gridap.CellData.DomainContribution, ::Nothing)
b
end
    
 


order = 1
reffeφ = ReferenceFE(lagrangian, Float64, order)
Vφ = TestFESpace(model, reffeφ,   conformity=:H1)
reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
Vu = TestFESpace(model, reffeu,   conformity=:H1)
V = MultiFieldFESpace([Vu, Vφ])

uh = FEFunction(Vu, 100.0.+ones(Float64, num_free_dofs(Vu)))
φh = FEFunction(Vφ, 100.0.+ones(Float64, num_free_dofs(Vφ)))

# using Mimosa:NeoHookean3D
# using Mimosa:IdealDielectric
# using Mimosa:ElectroMech
# using Mimosa:residual_M
# using Mimosa:DerivativeStrategy
# using Mimosa.WeakForms:CouplingStrategy
# using Mimosa.WeakForms:residual_EM

modmec = Mimosa.NeoHookean3D(λ=10.0, μ=1.0)
modelec = Mimosa.IdealDielectric(ε=1.0)
consmodel = Mimosa.ElectroMech(modmec, modelec)
ctype= Mimosa.WeakForms.CouplingStrategy{:monolithic}()
Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = consmodel(Mimosa.DerivativeStrategy{:analytic}())

degree = 2 * order
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

resa((u, φ), (v, vφ)) = Mimosa.WeakForms.residual_EM(ctype, (u, φ), (v, vφ), (∂Ψu, ∂Ψφ), dΩ) # Add Neumann BC 

 
function add_neumann(residual, bc::MultiFieldBoundaryCondition, dΓ;  Λ=1.0)
    res_neu((u, φ), (v, vφ)) = residual_Neumann_EM(ctype, (v, vφ), (bc.BoundaryCondition[1],bc.BoundaryCondition[2]), (dΓ[1],dΓ[2])) # Add Neumann BC 
    ((u, φ), (v, vφ))->residual((u, φ), (v, vφ))+ res_neu((u, φ), (v, vφ))
end


function add_neumann2(residual, bc::MultiFieldBoundaryCondition, dΓ;  Λ=1.0)
    res_neu((u, φ), (v, vφ)) = residual_Neumann(bc.BoundaryCondition[1], v, dΓ[1];  Λ=Λ)+residual_Neumann(bc.BoundaryCondition[2], vφ, dΓ[2];  Λ=Λ)
    ((u, φ), (v, vφ))->residual((u, φ), (v, vφ))+ res_neu((u, φ), (v, vφ))
end


rescc = add_neumann2(resa, neumannbc, dΓ;  Λ=1.0)

# res22=apply_neumann(res11, bc, dΓ;  Λ=1.0)

function res(uh, φh)
    r((v,vφ))=rescc((uh, φh),(v, vφ))
end

res2_=assemble_vector(res(uh, φh), V)
 
