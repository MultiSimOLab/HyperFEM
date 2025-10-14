module PhysicalModels

using Gridap
using Gridap.Helpers
using DrWatson

using ForwardDiff
using LinearAlgebra
using ..TensorAlgebra
using ..TensorAlgebra: _∂H∂F_2D
using StaticArrays

import Base: +

export Yeoh3D
export NeoHookean3D
export IncompressibleNeoHookean3D
export IncompressibleNeoHookean2D
export IncompressibleNeoHookean2D_CV
export IncompressibleNeoHookean3D_2dP
export ARAP2D
export ARAP2D_regularized
export VolumetricEnergy
export MooneyRivlin3D
export MooneyRivlin2D
export NonlinearMooneyRivlin3D
export NonlinearMooneyRivlin2D
export NonlinearMooneyRivlin2D_CV
export NonlinearNeoHookean_CV
export NonlinearMooneyRivlin_CV
export NonlinearIncompressibleMooneyRivlin2D_CV
export EightChain
export TransverseIsotropy3D
export TransverseIsotropy2D
export LinearElasticity3D
export LinearElasticity2D
export IdealDielectric
export IdealMagnetic
export IdealMagnetic2D
export HardMagnetic
export HardMagnetic2D
export ThermalModel
export ElectroMechModel
export ThermoElectroMechModel
export ThermoMechModel
export ThermoMech_EntropicPolyconvex
export FlexoElectroModel
export ThermoElectroMech_Govindjee
export ThermoElectroMech_PINNs
export ThermoElectroMech_Bonet
export MagnetoMechModel
export GeneralizedMaxwell
export ViscousIncompressible

export PhysicalModel
export Mechano
export Elasto
export Visco
export ViscoElastic
export Electro
export Magneto
export Thermo
export ElectroMechano
export MagnetoMechano
export ThermoElectroMechano
export ThermoMechano
export ThermoElectro
export FlexoElectro
export EnergyInterpolationScheme

export KinematicDescription
export DerivativeStrategy

export initializeStateVariables
export updateStateVariables!
export update_state!

export Kinematics
export KinematicModel
export EvolutiveKinematics
export get_Kinematics
export getIsoInvariants

export HessianRegularization
export Hessian∇JRegularization

struct DerivativeStrategy{Kind} end

abstract type PhysicalModel end
abstract type Mechano <: PhysicalModel end
abstract type Electro <: PhysicalModel end
abstract type Magneto <: PhysicalModel end
abstract type Thermo <: PhysicalModel end

abstract type Elasto <: Mechano end
abstract type Visco <: Mechano end
abstract type ViscoElastic <: Mechano end

abstract type MultiPhysicalModel <: PhysicalModel end
abstract type ElectroMechano <: MultiPhysicalModel end
abstract type ThermoElectroMechano <: MultiPhysicalModel end
abstract type ThermoMechano <: MultiPhysicalModel end
abstract type ThermoElectro <: MultiPhysicalModel end
abstract type FlexoElectro <: MultiPhysicalModel end
abstract type MagnetoMechano <: MultiPhysicalModel end

include("KinematicModels.jl")

include("MechanicalModels.jl")

include("ViscousModels.jl")

include("MagneticModels.jl")

include("ElectricalModels.jl")

include("ThermalModels.jl")

include("ElectroMechanicalModels.jl")

include("PINNs.jl")


# ============================================
# State variables management
# ============================================

function initializeStateVariables(::PhysicalModel, points::Measure)
  return nothing
end

function updateStateVariables!(::Any, ::PhysicalModel, vars...)
end



struct ThermoElectroMechModel{A,B,C} <: ThermoElectroMechano
  Thermo::A
  Electro::B
  Mechano::C
  fθ::Function
  dfdθ::Function
  function ThermoElectroMechModel(; Thermo::Thermo, Electro::Electro, Mechano::Mechano, fθ::Function, dfdθ::Function)
    A, B, C = typeof(Thermo), typeof(Electro), typeof(Mechano)
    new{A,B,C}(Thermo, Electro, Mechano, fθ, dfdθ)
  end


  function (obj::ThermoElectroMechModel)(Λ::Float64=1.0)
    Ψt, ∂Ψt_θ, ∂Ψt_θθ = obj.Thermo(Λ)
    Ψm, ∂Ψm_u, ∂Ψm_uu = obj.Mechano(Λ)
    Ψem, ∂Ψem_u, ∂Ψem_φ, ∂Ψem_uu, ∂Ψem_φu, ∂Ψem_φφ = _getCoupling(obj.Mechano, obj.Electro, Λ)
    Ψtm, ∂Ψtm_u, ∂Ψtm_θ, ∂Ψtm_uu, ∂Ψtm_uθ, ∂Ψtm_θθ = _getCoupling(obj.Mechano, obj.Thermo, Λ)
    f(δθ) = (obj.fθ(δθ)::Float64)
    df(δθ) = (obj.dfdθ(δθ)::Float64)

    Ψ(F, E, δθ) = f(δθ) * (Ψm(F) + Ψem(F, E)) + (Ψt(δθ) + Ψtm(F, δθ))
    ∂Ψu(F, E, δθ) = f(δθ) * (∂Ψm_u(F) + ∂Ψem_u(F, E)) + ∂Ψtm_u(F, δθ)
    ∂Ψφ(F, E, δθ) = f(δθ) * ∂Ψem_φ(F, E)
    ∂Ψθ(F, E, δθ) = df(δθ) * (Ψm(F) + Ψem(F, E)) + ∂Ψtm_θ(F, δθ) + ∂Ψt_θ(δθ)

    ∂Ψuu(F, E, δθ) = f(δθ) * (∂Ψm_uu(F) + ∂Ψem_uu(F, E)) + ∂Ψtm_uu(F, δθ)
    ∂Ψφu(F, E, δθ) = f(δθ) * ∂Ψem_φu(F, E)
    ∂Ψφφ(F, E, δθ) = f(δθ) * ∂Ψem_φφ(F, E)
    ∂Ψθθ(F, E, δθ) = ∂Ψtm_θθ(F, δθ) + ∂Ψt_θθ(δθ)
    ∂Ψuθ(F, E, δθ) = df(δθ) * (∂Ψm_u(F) + ∂Ψem_u(F, E)) + ∂Ψtm_uθ(F, δθ)
    ∂Ψφθ(F, E, δθ) = df(δθ) * ∂Ψem_φ(F, E)
    η(F, E, δθ) = -∂Ψθ(F, E, δθ)
    return (Ψ, ∂Ψu, ∂Ψφ, ∂Ψθ, ∂Ψuu, ∂Ψφφ, ∂Ψθθ, ∂Ψφu, ∂Ψuθ, ∂Ψφθ, η)
  end
end

struct ThermoElectroMech_Govindjee{A,B,C} <: ThermoElectroMechano
  Thermo::A
  Electro::B
  Mechano::C
  fθ::Function
  dfdθ::Function
  gθ::Function
  dgdθ::Function
  β::Float64
  function ThermoElectroMech_Govindjee(; Thermo::Thermo, Electro::Electro, Mechano::Mechano, fθ::Function, dfdθ::Function, gθ::Function, dgdθ::Function, β::Float64=0.0)
    A, B, C, = typeof(Thermo), typeof(Electro), typeof(Mechano)
    new{A,B,C}(Thermo, Electro, Mechano, fθ, dfdθ, gθ, dgdθ, β)
  end

  function (obj::ThermoElectroMech_Govindjee)(Λ::Float64=1.0)
    Ψm, _, _ = obj.Mechano(Λ)
    Ψem, _, _, _, _, _ = _getCoupling(obj.Mechano, obj.Electro, Λ)
    f(δθ) = obj.fθ(δθ)
    df(δθ) = obj.dfdθ(δθ)
    g(δθ) = obj.gθ(δθ)
    dg(δθ) = obj.dgdθ(δθ)

    _, _, J = get_Kinematics(obj.Mechano.Kinematic; Λ=Λ)
    Ψer(F) = obj.Thermo.α * (J(F) - 1.0) * obj.Thermo.θr
    ΨL1(δθ) = obj.Thermo.Cv * obj.Thermo.θr * (1 - obj.β) * ((δθ + obj.Thermo.θr) / obj.Thermo.θr * (1.0 - log((δθ + obj.Thermo.θr) / obj.Thermo.θr)) - 1.0)
    ΨL3(δθ) = g(δθ) - g(0.0) - dg(0.0) * δθ

    Ψ(F, E, δθ) = f(δθ) * (Ψm(F) + Ψem(F, E)) + (1 - f(δθ)) * Ψer(F) + ΨL1(δθ) + ΨL3(δθ) * (Ψm(F) + Ψem(F, E))
    ∂Ψ_∂F(F, E, θ) = ForwardDiff.gradient(F -> Ψ(F, get_array(E), θ), get_array(F))
    ∂Ψ_∂E(F, E, θ) = ForwardDiff.gradient(E -> Ψ(get_array(F), E, θ), get_array(E))
    ∂Ψ_∂θ(F, E, θ) = ForwardDiff.derivative(θ -> Ψ(get_array(F), get_array(E), θ), θ)

    ∂Ψu(F, E, θ) = TensorValue(∂Ψ_∂F(F, E, θ))
    ∂ΨE(F, E, θ) = VectorValue(∂Ψ_∂E(F, E, θ))
    ∂Ψθ(F, E, θ) = ∂Ψ_∂θ(F, E, θ)

    ∂2Ψ_∂2E(F, E, θ) = ForwardDiff.hessian(E -> Ψ(get_array(F), E, θ), get_array(E))
    ∂ΨEE(F, E, θ) = TensorValue(∂2Ψ_∂2E(F, E, θ))
    ∂2Ψθθ(F, E, θ) = ForwardDiff.derivative(θ -> ∂Ψ_∂θ(get_array(F), get_array(E), θ), θ)

    ∂2Ψ_∂2Eθ(F, E, θ) = ForwardDiff.derivative(θ -> ∂Ψ_∂E(get_array(F), get_array(E), θ), θ)
    ∂ΨEθ(F, E, θ) = VectorValue(∂2Ψ_∂2Eθ(F, E, θ))

    ∂2Ψ_∂2F(F, E, θ) = ForwardDiff.hessian(F -> Ψ(F, get_array(E), θ), get_array(F))
    ∂ΨFF(F, E, θ) = TensorValue(∂2Ψ_∂2F(F, E, θ))

    ∂2Ψ_∂2Fθ(F, E, θ) = ForwardDiff.derivative(θ -> ∂Ψ_∂F(get_array(F), get_array(E), θ), θ)
    ∂ΨFθ(F, E, θ) = TensorValue(∂2Ψ_∂2Fθ(F, E, θ))

    ∂2Ψ_∂EF(F, E, θ) = ForwardDiff.jacobian(F -> ∂Ψ_∂E(F, get_array(E), θ), get_array(F))
    ∂ΨEF(F, E, θ) = TensorValue(∂2Ψ_∂EF(F, E, θ))

    η(F, E, θ) = -∂Ψθ(F, E, θ)

    return (Ψ, ∂Ψu, ∂ΨE, ∂Ψθ, ∂ΨFF, ∂ΨEE, ∂2Ψθθ, ∂ΨEF, ∂ΨFθ, ∂ΨEθ, η)

  end
end


struct ThermoElectroMech_Bonet{A,B,C} <: ThermoElectroMechano
  Thermo::A
  Electro::B
  Mechano::C
  function ThermoElectroMech_Bonet(; Thermo::ThermalModel, Electro::Electro, Mechano::Mechano)
    A, B, C, = typeof(Thermo), typeof(Electro), typeof(Mechano)
    new{A,B,C}(Thermo, Electro, Mechano)
  end


  function (obj::ThermoElectroMech_Bonet)(Λ::Float64=1.0)
   @unpack Cv,θr, α, κ, γv, γd =obj.Thermo
    Ψem, ∂Ψem∂F, ∂Ψem∂E, ∂Ψem∂FF, ∂Ψem∂EF, ∂Ψem∂EE = _getCoupling(obj.Mechano, obj.Electro, Λ)
    gd(δθ) = 1/(γd+1) * (((δθ+θr)/θr)^(γd+1) -1)
    ∂gd(δθ) = (δθ+θr)^γd / θr^(γd+1)
    ∂∂gd(δθ) = γd*(δθ+θr)^(γd-1) / θr^(γd+1)
    gv(δθ) = 1/(γv+1) * (((δθ+θr)/θr)^(γv+1) -1)
    ∂gv(δθ) = (δθ+θr)^γv / θr^(γv+1)
    ∂∂gv(δθ) = γv*(δθ+θr)^(γv-1) / θr^(γv+1)

    _, H, J = get_Kinematics(obj.Mechano.Kinematic; Λ=Λ)

    η(F)=α*(J(F) - 1.0)+Cv/γv
    ∂η∂J(F)=α
    ∂η∂F(F)=∂η∂J(F)*H(F)
    ∂2η∂FF(F)=×ᵢ⁴(∂η∂J(F) * F)

    Ψ(F,E,δθ) = Ψem(F,E)*(1.0+gd(δθ))+gv(δθ)*η(F)

    ∂Ψ_∂F(F, E, δθ)  =   (1.0+gd(δθ)) *∂Ψem∂F(F, E) + gv(δθ)*∂η∂F(F)
    ∂Ψ_∂E(F, E, δθ)  =   (1.0+gd(δθ)) *∂Ψem∂E(F, E)
    ∂Ψ_∂δθ(F, E, δθ) =   ∂gd(δθ) *Ψem(F, E) + ∂gv(δθ)*η(F)

    ∂2Ψ_∂2F(F, E, δθ) =  (1.0+gd(δθ)) *∂Ψem∂FF(F, E) + gv(δθ)*∂2η∂FF(F)
    ∂2Ψ_∂2E(F, E, δθ) =  (1.0+gd(δθ)) *∂Ψem∂EE(F, E)
    ∂2Ψ_∂2δθ(F, E, δθ) =  ∂∂gd(δθ) *Ψem(F, E) + ∂∂gv(δθ)*η(F)

    ∂ΨEF(F, E, δθ) =  (1.0+gd(δθ)) *∂Ψem∂EF(F, E)
    ∂ΨFδθ(F, E, δθ) =  ∂gd(δθ) *∂Ψem∂F(F, E) + ∂gv(δθ)*∂η∂F(F)
    ∂ΨEδθ(F, E, δθ) =  ∂gd(δθ) *∂Ψem∂E(F, E)

    η(F, E, δθ) = -∂Ψ_∂δθ(F, E, δθ)

    return (Ψ, ∂Ψ_∂F, ∂Ψ_∂E, ∂Ψ_∂δθ, ∂2Ψ_∂2F, ∂2Ψ_∂2E, ∂2Ψ_∂2δθ, ∂ΨEF, ∂ΨFδθ, ∂ΨEδθ, η)
  end
end


end