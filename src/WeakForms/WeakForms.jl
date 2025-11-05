module WeakForms

using Gridap
using HyperFEM.TensorAlgebra
using HyperFEM.PhysicalModels
using HyperFEM.PhysicalModels:ViscoElectricModel

export residual
export jacobian
export mass_term
export (+)

import Gridap.Algebra:residual,jacobian
import Base: +

# ===================
# Coupling management
# ===================

function (+)(::Nothing, b::Gridap.CellData.DomainContribution)
b
end
function (+)(b::Gridap.CellData.DomainContribution, ::Nothing)
b
end
function (+)(a::Nothing, ::Nothing)
    a
end

# ===================
# Mechanics
# ===================

"""
    residual(...)::Gridap.CellData.Integrand

Calculate the residual using the given constitutive model and finite element functions.
"""
function residual(physicalmodel::Mechano, u, v, dΩ, Λ=1.0, Δt=0.0, vars...)
  _, ∂Ψu, _ = physicalmodel(Λ)
  F, _, _   = get_Kinematics(physicalmodel.Kinematic; Λ=Λ)
  ∫(∇(v)' ⊙ (∂Ψu ∘ (F∘∇(u)')))dΩ
end

"""
    jacobian(...)::Gridap.CellData.Integrand

Calculate the jacobian using the given constitutive model and finite element functions.
"""
function jacobian(physicalmodel::Mechano, u, du, v, dΩ, Λ=1.0, Δt=0.0, vars...)
  _, _, ∂Ψuu = physicalmodel(Λ)
  F, _, _    = get_Kinematics(physicalmodel.Kinematic; Λ=Λ)
  ∫(∇(v)' ⊙ ((∂Ψuu ∘ (F∘∇(u)')) ⊙ ∇(du)'))dΩ
end

function residual(physicalmodel::ViscoElastic, u, v, dΩ, t, Δt, un, A)
  _, ∂Ψu, _ = physicalmodel(t, Δt=Δt)
  F, _, _   = get_Kinematics(physicalmodel.Kinematic, Λ=t)
  ∫(∇(v)' ⊙ (∂Ψu ∘ (F∘∇(u)', F∘∇(un)', A...)))dΩ
end

function jacobian(physicalmodel::ViscoElastic, u, du, v, dΩ, t, Δt, un, A)
  _, _, ∂Ψuu = physicalmodel(t, Δt=Δt)
  F, _, _   = get_Kinematics(physicalmodel.Kinematic, Λ=t)
  ∫(∇(v)' ⊙ (inner ∘ (∂Ψuu∘(F∘∇(u)', F∘∇(un)', A...), ∇(du)')))dΩ
end


# ===================
# Mass term
# ===================

function mass_term(u, v, Coeff, dΩ)
    ∫(Coeff* (u⋅v))dΩ
end


# ===================
# ThermoElectroMech
# ===================

# Stagered strategy
# -----------------
function residual(physicalmodel::ThermoElectroMechano, ::Type{Mechano}, (u, φ, θ), v, dΩ, Λ=1.0)
    DΨ= physicalmodel(Λ)
    F,_,_ = get_Kinematics(physicalmodel.mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.electro.Kinematic; Λ=Λ)
    ∂Ψu=DΨ[2]
    return ∫(∇(v)' ⊙ (∂Ψu ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)))dΩ
end

function residual(physicalmodel::ThermoElectroMechano, ::Type{Electro}, (u, φ, θ), vφ, dΩ, Λ=1.0)
    DΨ= physicalmodel(Λ)
    F,_,_ = get_Kinematics(physicalmodel.mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.electro.Kinematic; Λ=Λ)
    ∂Ψφ=DΨ[3]
    return -1.0*∫(∇(vφ)' ⋅ (∂Ψφ ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)))dΩ
end

function residual(physicalmodel::ThermoElectroMechano, ::Type{Thermo}, (u, φ, θ), vθ, dΩ, Λ=1.0)
    κ=physicalmodel.thermo.κ
    return ∫(κ * ∇(θ) ⋅ ∇(vθ))dΩ
end


function jacobian(physicalmodel::ThermoElectroMechano, ::Type{Mechano}, (u, φ, θ), du, v, dΩ, Λ=1.0)
    DΨ= physicalmodel(Λ)
    F,_,_ = get_Kinematics(physicalmodel.mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.electro.Kinematic; Λ=Λ)
    ∂Ψuu=DΨ[5]
    ∫(∇(v)' ⊙ ((∂Ψuu ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) ⊙ (∇(du)')))dΩ
end

function jacobian(physicalmodel::ThermoElectroMechano, ::Type{Electro}, (u, φ, θ), dφ, vφ, dΩ, Λ=1.0)
    DΨ= physicalmodel(Λ)
    F,_,_ = get_Kinematics(physicalmodel.mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.electro.Kinematic; Λ=Λ)
    ∂Ψφφ=DΨ[6]
    ∫(∇(vφ) ⋅ ((∂Ψφφ ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) ⋅ ∇(dφ)))dΩ
end

function jacobian(physicalmodel::ThermoElectroMechano, ::Type{Thermo}, dθ, vθ, dΩ, Λ=1.0)
    κ=physicalmodel.thermo.κ
    ∫(κ * ∇(dθ) ⋅ ∇(vθ))dΩ
end

function jacobian(physicalmodel::ThermoElectroMechano, ::Type{Thermo}, (u, φ, θ)::Tuple, dθ, vθ, dΩ, Λ=1.0)
    κ=physicalmodel.thermo.κ
    ∫((κ ∘ (u, φ, θ)) * ∇(dθ) ⋅ ∇(vθ))dΩ
end

function jacobian(physicalmodel::ThermoElectroMechano, ::Type{ElectroMechano}, (u, φ, θ), (du, dφ), (v,vφ), dΩ, Λ=1.0)
    DΨ= physicalmodel(Λ)
    F,_,_ = get_Kinematics(physicalmodel.mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.electro.Kinematic; Λ=Λ)
    ∂Ψφu=DΨ[8]
    -1.0*∫(∇(dφ) ⋅ ((∂Ψφu ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) ⊙ (∇(v)')))dΩ -
    ∫(∇(vφ) ⋅ ((∂Ψφu ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) ⊙ (∇(du)')))dΩ 
end

function jacobian(physicalmodel::ThermoElectroMechano, ::Type{ThermoMechano}, (u, φ, θ), dθ, v, dΩ, Λ=1.0)
    DΨ= physicalmodel(Λ)
    F,_,_ = get_Kinematics(physicalmodel.mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.electro.Kinematic; Λ=Λ)
    ∂Ψuθ=DΨ[9]
    ∫(∇(v)' ⊙ (∂Ψuθ ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) * dθ)dΩ 
end

function jacobian(physicalmodel::ThermoElectroMechano, ::Type{ThermoElectro}, (u, φ, θ), dθ, vφ, dΩ, Λ=1.0)
    DΨ= physicalmodel(Λ)
    F,_,_ = get_Kinematics(physicalmodel.mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.electro.Kinematic; Λ=Λ)
    ∂Ψφθ=DΨ[10]
    -1.0*∫(∇(vφ) ⋅ ((∂Ψφθ ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) * dθ))dΩ
end

# Monolithic strategy
# -------------------
function residual(physicalmodel::ThermoElectroMechano, (u, φ, θ), (v, vφ, vθ), dΩ, Λ=1.0)
    residual(physicalmodel, Mechano, (u, φ, θ), v, dΩ, Λ) +
    residual(physicalmodel, Electro, (u, φ, θ), vφ, dΩ, Λ) +
    residual(physicalmodel, Thermo, (u, φ, θ), vθ, dΩ, Λ)
end

function jacobian(physicalmodel::ThermoElectroMechano, (u, φ, θ), (du, dφ, dθ), (v, vφ, vθ),  dΩ, Λ=1.0)
    jacobian(physicalmodel, Mechano, (u, φ, θ), du, v, dΩ, Λ)+
    jacobian(physicalmodel, Electro, (u, φ, θ), dφ, vφ, dΩ, Λ)+
    jacobian(physicalmodel, Thermo, dθ, vθ, dΩ, Λ)+
    jacobian(physicalmodel, ElectroMechano, (u, φ, θ), (du, dφ), (v,vφ), dΩ, Λ)+
    jacobian(physicalmodel, ThermoMechano, (u, φ, θ), dθ, v, dΩ, Λ)+
    jacobian(physicalmodel, ThermoElectro, (u, φ, θ), dθ, vφ, dΩ, Λ)
end


# ===================
# ThermoMech
# ===================

# Stagered strategy
# -----------------
function residual(physicalmodel::ThermoMechano, ::Type{Mechano}, (u, θ), v, dΩ, Λ=1.0)
    DΨ= physicalmodel(Λ)
    F,_,_ = get_Kinematics(physicalmodel.mechano.Kinematic; Λ=Λ)
    ∂Ψu=DΨ[2]
    ∫(∇(v)' ⊙ (∂Ψu ∘ (F∘(∇(u)'), θ)))dΩ
end

function residual(physicalmodel::ThermoMechano, ::Type{Thermo}, (u, θ), vθ, dΩ, Λ=1.0)
    κ=physicalmodel.thermo.κ
    ∫(κ * ∇(θ) ⋅ ∇(vθ))dΩ
end

function jacobian(physicalmodel::ThermoMechano, ::Type{Mechano}, (u, θ), du, v, dΩ, Λ=1.0)
    DΨ= physicalmodel(Λ)
    F,_,_ = get_Kinematics(physicalmodel.mechano.Kinematic; Λ=Λ)
    ∂Ψuu=DΨ[4]
    ∫(∇(v)' ⊙ ((∂Ψuu ∘ (F∘(∇(u)'), θ)) ⊙ (∇(du)')))dΩ
end

function jacobian(physicalmodel::ThermoMechano, ::Type{Thermo}, dθ, vθ, dΩ, Λ=1.0)
    κ=physicalmodel.thermo.κ
    ∫(κ * ∇(dθ) ⋅ ∇(vθ))dΩ
end

function jacobian(physicalmodel::ThermoMechano, ::Type{Thermo}, (u, θ)::Tuple, dθ, vθ, dΩ, Λ=1.0)
    κ=physicalmodel.thermo.κ
    ∫((κ ∘ (u, θ)) * ∇(dθ) ⋅ ∇(vθ))dΩ
end

function jacobian(physicalmodel::ThermoMechano, ::Type{ThermoMechano}, (u, θ), (du, dθ), v, dΩ, Λ=1.0)
    DΨ= physicalmodel(Λ)
    F,_,_ = get_Kinematics(physicalmodel.mechano.Kinematic; Λ=Λ)
    ∂Ψuθ=DΨ[6]
    ∫(∇(v)' ⊙ (∂Ψuθ ∘ (F∘(∇(u)'), θ)) * dθ)dΩ 
end

# Monolithic strategy
# -------------------
function residual(physicalmodel::ThermoMechano,  (u, θ), (v, vθ), dΩ, Λ=1.0)
    residual(physicalmodel, Mechano, (u, θ), v, dΩ, Λ) +
    residual(physicalmodel, Thermo, (u, θ), vθ, dΩ, Λ)
end

function jacobian(physicalmodel::ThermoMechano,  (u, θ), (du, dθ), (v, vθ),  dΩ, Λ=1.0)
    jacobian(physicalmodel, Mechano, (u, θ), du, v, dΩ, Λ)+
    jacobian(physicalmodel, Thermo, dθ, vθ, dΩ, Λ)+
    jacobian(physicalmodel, ThermoMechano, (u, θ), (du, dθ), v, dΩ, Λ)
end

# ===================
# ElectroMechanics
# ===================

include("ElectroMechanics.jl")


# =====================
# FlexoElectroMechanics
# =====================

# Stagered strategy
# -----------------
 

function residual(physicalmodel::FlexoElectro, ::Type{Mechano}, (u, φ, ϕ₁, ϕ₂, ϕ₃), v, dΩ, X, Λ=1.0)
    DΨ= physicalmodel(Λ)
    κ=physicalmodel.κ
    F,_,_ = get_Kinematics(physicalmodel.electromechano.mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.electromechano.electro.Kinematic; Λ=Λ)
    ∂Ψu=DΨ[2]
    Φ=DΨ[7]
    ∫((∇(v)' ⊙ (∂Ψu ∘ (F∘(∇(u)',X), E∘(∇(φ)))+κ*(∇(u)'-(Φ(ϕ₁,ϕ₂,ϕ₃))))))dΩ
end

function residual(physicalmodel::FlexoElectro, ::Type{Electro}, (u, φ), vφ, dΩ, X, Λ=1.0)
    DΨ= physicalmodel(Λ)
    F,_,_ = get_Kinematics(physicalmodel.electromechano.mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.electromechano.Electro.Kinematic; Λ=Λ)
    ∂Ψφ=DΨ[3]
    -1.0*∫((∇(vφ) ⋅ (∂Ψφ ∘ (F∘(∇(u)',X), E∘(∇(φ))))))dΩ
end

function residual(physicalmodel::FlexoElectro, ::Type{FlexoElectro}, (u, ϕ₁, ϕ₂, ϕ₃), (δϕ₁, δϕ₂, δϕ₃), dΩ, X, Λ=1.0)
    DΨ= physicalmodel(Λ)
    κ=physicalmodel.κ
    Φ=DΨ[7]
    ∫((κ⋅(-∇(u)'+(Φ(ϕ₁,ϕ₂,ϕ₃)))) ⊙ (Φ(δϕ₁,δϕ₂,δϕ₃)))dΩ
end

function jacobian(physicalmodel::FlexoElectro, ::Type{Mechano}, (u, φ), du, v, dΩ, X, Λ=1.0)
    DΨ= physicalmodel(Λ)
    κ=physicalmodel.κ
    F,_,_ = get_Kinematics(physicalmodel.electroMechano.mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.electroMechano.electro.Kinematic; Λ=Λ)
    ∂Ψuu=DΨ[4]
    ∫(∇(v)' ⊙ ((∂Ψuu ∘ (F∘(∇(u)',X), E∘(∇(φ)))) ⊙ (∇(du)')))dΩ+
    ∫(κ⋅(∇(v)'⊙ ∇(du)'))dΩ
end

function jacobian(physicalmodel::FlexoElectro, ::Type{Electro}, (u, φ), dφ, vφ, dΩ, X, Λ=1.0)
    DΨ= physicalmodel(Λ)
    F,_,_ = get_Kinematics(physicalmodel.electromechano.mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.electromechano.electro.Kinematic; Λ=Λ)
    ∂Ψφφ=DΨ[6]
    ∫(∇(vφ)' ⋅ ((∂Ψφφ ∘ (F∘(∇(u)',X), E∘(∇(φ)))) ⋅ ∇(dφ)))dΩ
end

function jacobian(physicalmodel::FlexoElectro, ::Type{ElectroMechano}, (u, φ), (du, dφ), (v, vφ), dΩ, X, Λ=1.0)
    DΨ= physicalmodel(Λ)
    F,_,_ = get_Kinematics(physicalmodel.electromechano.mechano.Kinematic; Λ=Λ)
    E     = get_Kinematics(physicalmodel.electromechano.electro.Kinematic; Λ=Λ)
    ∂Ψφu=DΨ[5]
    -1.0*∫(∇(dφ) ⋅ ((∂Ψφu ∘ (F∘(∇(u)',X), E∘(∇(φ)))) ⊙ (∇(v)')))dΩ -
    ∫(∇(vφ) ⋅ ((∂Ψφu ∘ (F∘(∇(u)',X), E∘(∇(φ)))) ⊙ (∇(du)')))dΩ 
end


function jacobian(physicalmodel::FlexoElectro, ::Type{FlexoElectro}, (ϕ₁,ϕ₂,ϕ₃), (du, dϕ₁,dϕ₂,dϕ₃), (v, δϕ₁,δϕ₂,δϕ₃), dΩ, X, Λ=1.0)
    DΨ= physicalmodel(Λ)
    κ=physicalmodel.κ
    Φ=DΨ[7]
    ∫((κ⋅(Φ(δϕ₁,δϕ₂,δϕ₃))) ⊙ ((Φ(dϕ₁,dϕ₂,dϕ₃))))dΩ-
    ∫(∇(v)' ⊙ (κ*(Φ(dϕ₁,dϕ₂,dϕ₃))))dΩ-
    ∫(∇(du)' ⊙ (κ*(Φ(δϕ₁,δϕ₂,δϕ₃))))dΩ
end


# Monolithic strategy
# -------------------

function residual(physicalmodel::FlexoElectro, (u, φ, ϕ₁, ϕ₂, ϕ₃), (v, vφ, δϕ₁, δϕ₂, δϕ₃), dΩ, X, Λ=1.0)
    residual(physicalmodel, Mechano, (u, φ, ϕ₁, ϕ₂, ϕ₃), v, dΩ, X, Λ) +
    residual(physicalmodel, Electro, (u, φ), vφ, dΩ, X, Λ)+
    residual(physicalmodel, FlexoElectro, (u, ϕ₁, ϕ₂, ϕ₃), (δϕ₁, δϕ₂, δϕ₃), dΩ, X, Λ)
end

function jacobian(physicalmodel::FlexoElectro, (u, φ, ϕ₁,ϕ₂,ϕ₃), (du, dφ, dϕ₁,dϕ₂,dϕ₃), (v, vφ, δϕ₁,δϕ₂,δϕ₃), dΩ, X, Λ=1.0)
    jacobian(physicalmodel, Mechano, (u, φ), du, v, dΩ, X, Λ)+
    jacobian(physicalmodel, Electro, (u, φ), dφ, vφ, dΩ, X, Λ)+
    jacobian(physicalmodel, ElectroMechano, (u, φ), (du, dφ), (v, vφ), dΩ, X, Λ)+
    jacobian(physicalmodel, FlexoElectro, (ϕ₁,ϕ₂,ϕ₃), (du, dϕ₁,dϕ₂,dϕ₃), (v, δϕ₁,δϕ₂,δϕ₃), dΩ, X, Λ)
end





# ===================
# ThermoElectroMech_PINNs
# ===================

# Stagered strategy
# -----------------
function residual(physicalmodel::ThermoElectroMech_PINNs, ::Type{Mechano}, (u, φ, θ), v, dΩ, Λ=1.0)
    DΨ= physicalmodel(Λ)
    Kinematic_mec = Kinematics(Mechano)
    Kinematic_elec = Kinematics(Electro)
    F, _, _ = get_Kinematics(Kinematic_mec)
    E = get_Kinematics(Kinematic_elec)
    ∂Ψu=DΨ[2]    
    return ∫(∇(v)' ⊙ (∂Ψu ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)))dΩ
end

function residual(physicalmodel::ThermoElectroMech_PINNs, ::Type{Electro}, (u, φ, θ), vφ, dΩ, Λ=1.0)
    DΨ= physicalmodel(Λ)
    Kinematic_mec = Kinematics(Mechano)
    Kinematic_elec = Kinematics(Electro)
    F, _, _ = get_Kinematics(Kinematic_mec)
    E = get_Kinematics(Kinematic_elec)
    ∂Ψφ=DΨ[3]
    return -1.0*∫(∇(vφ)' ⋅ (∂Ψφ ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)))dΩ
end

function residual(physicalmodel::ThermoElectroMech_PINNs, ::Type{Thermo}, (u, φ, θ), vθ, dΩ, Λ=1.0)
    κ=physicalmodel.κ
    return ∫(κ * ∇(θ) ⋅ ∇(vθ))dΩ
end


function jacobian(physicalmodel::ThermoElectroMech_PINNs, ::Type{Mechano}, (u, φ, θ), du, v, dΩ, Λ=1.0)
    DΨ= physicalmodel(Λ)
    Kinematic_mec = Kinematics(Mechano)
    Kinematic_elec = Kinematics(Electro)
    F, _, _ = get_Kinematics(Kinematic_mec)
    E = get_Kinematics(Kinematic_elec)
    ∂Ψuu=DΨ[5]
    ∫(∇(v)' ⊙ ((∂Ψuu ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) ⊙ (∇(du)')))dΩ
end

function jacobian(physicalmodel::ThermoElectroMech_PINNs, ::Type{Electro}, (u, φ, θ), dφ, vφ, dΩ, Λ=1.0)
    DΨ= physicalmodel(Λ)
    Kinematic_mec = Kinematics(Mechano)
    Kinematic_elec = Kinematics(Electro)
    F, _, _ = get_Kinematics(Kinematic_mec)
    E = get_Kinematics(Kinematic_elec)
    ∂Ψφφ=DΨ[6]
    ∫(∇(vφ) ⋅ ((∂Ψφφ ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) ⋅ ∇(dφ)))dΩ
end

function jacobian(physicalmodel::ThermoElectroMech_PINNs, ::Type{Thermo}, dθ, vθ, dΩ, Λ=1.0)
    κ=physicalmodel.κ
    ∫(κ * ∇(dθ) ⋅ ∇(vθ))dΩ
end

function jacobian(physicalmodel::ThermoElectroMech_PINNs, ::Type{Thermo}, (u, φ, θ)::Tuple, dθ, vθ, dΩ, Λ=1.0)
    κ=physicalmodel.κ
    ∫((κ ∘ (u, φ, θ)) * ∇(dθ) ⋅ ∇(vθ))dΩ
end

function jacobian(physicalmodel::ThermoElectroMech_PINNs, ::Type{ElectroMechano}, (u, φ, θ), (du, dφ), (v,vφ), dΩ, Λ=1.0)
    DΨ= physicalmodel(Λ)
    Kinematic_mec = Kinematics(Mechano)
    Kinematic_elec = Kinematics(Electro)
    F, _, _ = get_Kinematics(Kinematic_mec)
    E = get_Kinematics(Kinematic_elec)
    ∂Ψφu=DΨ[8]
    -1.0*∫(∇(dφ) ⋅ ((∂Ψφu ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) ⊙ (∇(v)')))dΩ -
    ∫(∇(vφ) ⋅ ((∂Ψφu ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) ⊙ (∇(du)')))dΩ 
end

function jacobian(physicalmodel::ThermoElectroMech_PINNs, ::Type{ThermoMechano}, (u, φ, θ), dθ, v, dΩ, Λ=1.0)
    DΨ= physicalmodel(Λ)
    Kinematic_mec = Kinematics(Mechano)
    Kinematic_elec = Kinematics(Electro)
    F, _, _ = get_Kinematics(Kinematic_mec)
    E = get_Kinematics(Kinematic_elec)
    ∂Ψuθ=DΨ[9]
    ∫(∇(v)' ⊙ (∂Ψuθ ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) * dθ)dΩ 
end

function jacobian(physicalmodel::ThermoElectroMech_PINNs, ::Type{ThermoElectro}, (u, φ, θ), dθ, vφ, dΩ, Λ=1.0)
    DΨ= physicalmodel(Λ)
    Kinematic_mec = Kinematics(Mechano)
    Kinematic_elec = Kinematics(Electro)
    F, _, _ = get_Kinematics(Kinematic_mec)
    E = get_Kinematics(Kinematic_elec)
    ∂Ψφθ=DΨ[10]
    -1.0*∫(∇(vφ) ⋅ ((∂Ψφθ ∘ (F∘(∇(u)'), E∘(∇(φ)), θ)) * dθ))dΩ
end

# Monolithic strategy
# -------------------
function residual(physicalmodel::ThermoElectroMech_PINNs, (u, φ, θ), (v, vφ, vθ), dΩ, Λ=1.0)
    residual(physicalmodel, Mechano, (u, φ, θ), v, dΩ, Λ) +
    residual(physicalmodel, Electro, (u, φ, θ), vφ, dΩ, Λ) +
    residual(physicalmodel, Thermo, (u, φ, θ), vθ, dΩ, Λ)
end

function jacobian(physicalmodel::ThermoElectroMech_PINNs, (u, φ, θ), (du, dφ, dθ), (v, vφ, vθ),  dΩ, Λ=1.0)
    jacobian(physicalmodel, Mechano, (u, φ, θ), du, v, dΩ, Λ)+
    jacobian(physicalmodel, Electro, (u, φ, θ), dφ, vφ, dΩ, Λ)+
    jacobian(physicalmodel, Thermo, dθ, vθ, dΩ, Λ)+
    jacobian(physicalmodel, ElectroMechano, (u, φ, θ), (du, dφ), (v,vφ), dΩ, Λ)+
    jacobian(physicalmodel, ThermoMechano, (u, φ, θ), dθ, v, dΩ, Λ)+
    jacobian(physicalmodel, ThermoElectro, (u, φ, θ), dθ, vφ, dΩ, Λ)
end


end