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





struct MagnetoMechModel{A,B} <: MagnetoMechano
  Mechano::A
  Magneto::B
  function MagnetoMechModel(; Mechano::Mechano, Magneto::Magneto)
    A, B = typeof(Mechano), typeof(Magneto)
    new{A,B}(Mechano, Magneto)
  end

  function (obj::MagnetoMechModel)(Λ::Float64=1.0)
    Ψm, ∂Ψm_u, ∂Ψm_uu = obj.Mechano(Λ)
    Ψmm, ∂Ψmm_u, ∂Ψmm_φ, ∂Ψmm_uu, ∂Ψmm_φu, ∂Ψmm_φφ = _getCoupling(obj.Mechano, obj.Magneto, Λ)


    Ψ(F, ℋ₀, N) = Ψm(F) + Ψmm(F, ℋ₀, N)
    ∂Ψu(F, ℋ₀, N) = ∂Ψm_u(F) + ∂Ψmm_u(F, ℋ₀, N)
    ∂Ψφ(F, ℋ₀, N) = ∂Ψmm_φ(F, ℋ₀, N)
    ∂Ψuu(F, ℋ₀, N) = ∂Ψm_uu(F) + ∂Ψmm_uu(F, ℋ₀, N)
    ∂Ψφu(F, ℋ₀, N) = ∂Ψmm_φu(F, ℋ₀, N)
    ∂Ψφφ(F, ℋ₀, N) = ∂Ψmm_φφ(F, ℋ₀, N)

    return (Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ)
  end



end



# ===============================
# Coupling terms for multiphysic
# ===============================


function _getCoupling(::Mechano, mag::Union{IdealMagnetic,IdealMagnetic2D}, Λ::Float64)

  Ψmm, ∂Ψmm_∂u, ∂Ψmm_∂φ, ∂Ψmm_∂uu, ∂Ψmm_∂φu, ∂Ψmm_∂φφ = mag(Λ)

  Ψ(F, ℋ₀, N) = Ψmm(F, ℋ₀)
  ∂Ψ_u(F, ℋ₀, N) = ∂Ψmm_∂u(F, ℋ₀)
  ∂Ψ_φ(F, ℋ₀, N) = ∂Ψmm_∂φ(F, ℋ₀)
  ∂Ψ_uu(F, ℋ₀, N) = ∂Ψmm_∂uu(F, ℋ₀)
  ∂Ψ_φu(F, ℋ₀, N) = ∂Ψmm_∂φu(F, ℋ₀)
  ∂Ψ_φφ(F, ℋ₀, N) = ∂Ψmm_∂φφ(F, ℋ₀)


  return (Ψ, ∂Ψ_u, ∂Ψ_φ, ∂Ψ_uu, ∂Ψ_φu, ∂Ψ_φφ)

end

function _getCoupling(mec::Mechano, mag::HardMagnetic, Λ::Float64)

  # Miguel Angel Moreno-Mateos, Mokarram Hossain, Paul Steinmann, Daniel Garcia-Gonzalez,
  # Hard magnetics in ultra-soft magnetorheological elastomers enhance fracture toughness and 
  # delay crack propagation, Journal of the Mechanics and Physics of Solids,

  _, H, J = get_Kinematics(mec.Kinematic; Λ=Λ)
  μ, αr, χe, χr, βcoup, βmok = mag.μ, mag.αr, mag.χe, mag.χr, mag.βcoup, mag.βmok

  αr *= Λ
  #-------------------------------------------------------------------------------------
  # FIRST TERM
  #-------------------------------------------------------------------------------------

  Ψmm, ∂Ψmm_∂u, ∂Ψmm_∂φ, ∂Ψmm_∂uu, ∂Ψmm_∂φu, ∂Ψmm_∂φφ = mag(Λ)


  #-------------------------------------------------------------------------------------
  # SECOND TERM
  #-------------------------------------------------------------------------------------
  Hℋ₀(F, ℋ₀) = H(F) * ℋ₀
  Hℋ₀Hℋ₀(F, ℋ₀) = Hℋ₀(F, ℋ₀) ⋅ Hℋ₀(F, ℋ₀)

  ℋᵣ(N) = αr * N
  Fℋᵣ(F, N) = F * ℋᵣ(N)
  Ψcoup(F, N) = (μ * J(F)) * (Fℋᵣ(F, N) ⋅ Fℋᵣ(F, N) - ℋᵣ(N) ⋅ ℋᵣ(N))
  ∂Ψcoup_∂F(F, N) = 2 * (μ * J(F)) * (Fℋᵣ(F, N) ⊗ ℋᵣ(N))
  ∂Ψcoup_∂J(F, N) = (μ) * (Fℋᵣ(F, N) ⋅ Fℋᵣ(F, N) - ℋᵣ(N) ⋅ ℋᵣ(N))
  ∂Ψcoup_∂u(F, N) = ∂Ψcoup_∂J(F, N) * H(F) + ∂Ψcoup_∂F(F, N)

  ∂Ψcoup_∂JF(F, N) = 2 * μ * (H(F) ⊗ (Fℋᵣ(F, N) ⊗ ℋᵣ(N)) + (Fℋᵣ(F, N) ⊗ ℋᵣ(N)) ⊗ H(F))
  ∂Ψcoup_∂FF(F, N) = 2 * μ * J(F) * (I3 ⊗₁₃²⁴ (ℋᵣ(N) ⊗ ℋᵣ(N)))
  ∂Ψcoup_∂uu(F, N) = ∂Ψcoup_∂JF(F, N) + ∂Ψcoup_∂FF(F, N) + ∂Ψcoup_∂J(F, N) * ×ᵢ⁴(F)

  #-------------------------------------------------------------------------------------
  # THIRD TERM
  #-------------------------------------------------------------------------------------

  Ψmok(F, N) = (0.5 * μ * J(F) / χr) * (ℋᵣ(N) ⋅ ℋᵣ(N))
  ∂Ψmok_∂u(F, N) = (0.5 * μ / χr) * (ℋᵣ(N) ⋅ ℋᵣ(N)) * H(F)
  ∂Ψmok_∂uu(F, N) = (0.5 * μ / χr) * (ℋᵣ(N) ⋅ ℋᵣ(N)) * ×ᵢ⁴(F)

  #-------------------------------------------------------------------------------------
  # FOURTH TERM
  #-------------------------------------------------------------------------------------
  Hℋᵣ(F, N) = H(F) * ℋᵣ(N)
  Ψtorq(F, ℋ₀, N) = (μ * (1 + χe) / J(F)) * (Hℋ₀(F, ℋ₀) ⋅ Hℋᵣ(F, N))
  ∂Ψtorq_∂H(F, ℋ₀, N) = (μ * (1 + χe) / J(F)) * (Hℋᵣ(F, N) ⊗ ℋ₀ + Hℋ₀(F, ℋ₀) ⊗ ℋᵣ(N))
  ∂Ψtorq_∂J(F, ℋ₀, N) = -(μ * (1 + χe) / J(F)^2) * (Hℋ₀(F, ℋ₀) ⋅ Hℋᵣ(F, N))
  ∂Ψtorq_∂u(F, ℋ₀, N) = ∂Ψtorq_∂H(F, ℋ₀, N) × F + ∂Ψtorq_∂J(F, ℋ₀, N) * H(F)
  ∂Ψtorq_∂φ(F, ℋ₀, N) = (μ * (1 + χe) / J(F)) * (H(F)' * Hℋᵣ(F, N))

  ∂Ψtorq_∂HH(F, ℋ₀, N) = (μ * (1 + χe) / J(F)) * (I3 ⊗₁₃²⁴ (ℋᵣ(N) ⊗ ℋ₀ + ℋ₀ ⊗ ℋᵣ(N)))
  ∂Ψtorq_∂HJ(F, ℋ₀, N) = -(μ * (1 + χe) / J(F)^2) * (Hℋᵣ(F, N) ⊗ ℋ₀ + Hℋ₀(F, ℋ₀) ⊗ ℋᵣ(N))
  ∂Ψtorq_∂JJ(F, ℋ₀, N) = (μ * (1 + χe) / J(F)^3) * (Hℋ₀(F, ℋ₀) ⋅ Hℋᵣ(F, N))

  ∂Ψtorq_∂uu(F, ℋ₀, N) = (F × (∂Ψtorq_∂HH(F, ℋ₀, N) × F)) +
                         H(F) ⊗₁₂³⁴ (∂Ψtorq_∂HJ(F, ℋ₀, N) × F) +
                         (∂Ψtorq_∂HJ(F, ℋ₀, N) × F) ⊗₁₂³⁴ H(F) +
                         ∂Ψtorq_∂JJ(F, ℋ₀, N) * (H(F) ⊗₁₂³⁴ H(F)) +
                         ×ᵢ⁴(∂Ψtorq_∂H(F, ℋ₀, N) + ∂Ψtorq_∂J(F, ℋ₀, N) * F)


  ∂Ψtorq_∂ℋ₀H(F, ℋ₀, N) = (μ / (J(F))) * ((I3 ⊗₁₃² Hℋᵣ(F, N)) + (H(F)' ⊗₁₂³ Hℋᵣ(F, N))) * (1 + χe)
  ∂Ψtorq_∂ℋ₀J(F, ℋ₀, N) = (-μ / (J(F))^2.0) * (H(F)' * Hℋᵣ(F, N)) * (1 + χe)


  ∂Ψtorq_∂φu(F, ℋ₀, N) = (∂Ψtorq_∂ℋ₀H(F, ℋ₀, N) × F) + (∂Ψtorq_∂ℋ₀J(F, ℋ₀, N) ⊗₁²³ H(F))


  #-------------------------------------------------------------------------------------
  #                           TOTAL ENERGY
  #-------------------------------------------------------------------------------------
  Ψ(F, ℋ₀, N) = Ψmm(F, ℋ₀) + βcoup * Ψcoup(F, N) + βmok * Ψmok(F, N) + Ψtorq(F, ℋ₀, N)
  ∂Ψ_u(F, ℋ₀, N) = ∂Ψmm_∂u(F, ℋ₀) + βcoup * ∂Ψcoup_∂u(F, N) + βmok * ∂Ψmok_∂u(F, N) + ∂Ψtorq_∂u(F, ℋ₀, N)
  ∂Ψ_φ(F, ℋ₀, N) = ∂Ψmm_∂φ(F, ℋ₀) + ∂Ψtorq_∂φ(F, ℋ₀, N)
  ∂Ψ_uu(F, ℋ₀, N) = ∂Ψmm_∂uu(F, ℋ₀) + βcoup * ∂Ψcoup_∂uu(F, N) + βmok * ∂Ψmok_∂uu(F, N) + ∂Ψtorq_∂uu(F, ℋ₀, N)
  ∂Ψ_φu(F, ℋ₀, N) = ∂Ψmm_∂φu(F, ℋ₀) + ∂Ψtorq_∂φu(F, ℋ₀, N)
  ∂Ψ_φφ(F, ℋ₀, N) = ∂Ψmm_∂φφ(F, ℋ₀)

  return (Ψ, ∂Ψ_u, ∂Ψ_φ, ∂Ψ_uu, ∂Ψ_φu, ∂Ψ_φφ)

end

function _getCoupling(mec::Mechano, mag::HardMagnetic2D, Λ::Float64)

  # Miguel Angel Moreno-Mateos, Mokarram Hossain, Paul Steinmann, Daniel Garcia-Gonzalez,
  # Hard magnetics in ultra-soft magnetorheological elastomers enhance fracture toughness and 
  # delay crack propagation, Journal of the Mechanics and Physics of Solids,

  _, H, J = get_Kinematics(mec.Kinematic; Λ=Λ)
  μ, αr, χe, χr, βcoup, βmok = mag.μ, mag.αr, mag.χe, mag.χr, mag.βcoup, mag.βmok
  αr *= Λ


  # #-------------------------------------------------------------------------------------
  # # FIRST TERM
  # #-------------------------------------------------------------------------------------

  Ψmm, ∂Ψmm_∂u, ∂Ψmm_∂φ, ∂Ψmm_∂uu, ∂Ψmm_∂φu, ∂Ψmm_∂φφ = mag(Λ)


  #-------------------------------------------------------------------------------------
  # SECOND TERM
  #-------------------------------------------------------------------------------------
  Hℋ₀(F, ℋ₀) = H(F) * ℋ₀
  Hℋ₀Hℋ₀(F, ℋ₀) = Hℋ₀(F, ℋ₀) ⋅ Hℋ₀(F, ℋ₀)
  ℋᵣ(N) = αr * N
  Fℋᵣ(F, N) = F * ℋᵣ(N)
  Ψcoup(F, N) = (μ * J(F)) * (Fℋᵣ(F, N) ⋅ Fℋᵣ(F, N) - ℋᵣ(N) ⋅ ℋᵣ(N))
  ∂Ψcoup_∂F(F, N) = 2 * (μ * J(F)) * (Fℋᵣ(F, N) ⊗ ℋᵣ(N))
  ∂Ψcoup_∂J(F, N) = (μ) * (Fℋᵣ(F, N) ⋅ Fℋᵣ(F, N) - ℋᵣ(N) ⋅ ℋᵣ(N))
  ∂Ψcoup_∂u(F, N) = ∂Ψcoup_∂J(F, N) * H(F) + ∂Ψcoup_∂F(F, N)

  ∂Ψcoup_∂JF(F, N) = 2 * μ * (H(F) ⊗ (Fℋᵣ(F, N) ⊗ ℋᵣ(N)) + (Fℋᵣ(F, N) ⊗ ℋᵣ(N)) ⊗ H(F))
  ∂Ψcoup_∂FF(F, N) = 2 * μ * J(F) * (I2 ⊗₁₃²⁴ (ℋᵣ(N) ⊗ ℋᵣ(N)))
  ∂Ψcoup_∂uu(F, N) = ∂Ψcoup_∂JF(F, N) + ∂Ψcoup_∂FF(F, N) + ∂Ψcoup_∂J(F, N) * _∂H∂F_2D()

  #-------------------------------------------------------------------------------------
  # THIRD TERM
  #-------------------------------------------------------------------------------------

  Ψmok(F, N) = (0.5 * μ * J(F) / χr) * (ℋᵣ(N) ⋅ ℋᵣ(N))
  ∂Ψmok_∂u(F, N) = (0.5 * μ / χr) * (ℋᵣ(N) ⋅ ℋᵣ(N)) * H(F)
  ∂Ψmok_∂uu(F, N) = (0.5 * μ / χr) * (ℋᵣ(N) ⋅ ℋᵣ(N)) * _∂H∂F_2D()

  #-------------------------------------------------------------------------------------
  # FOURTH TERM
  #-------------------------------------------------------------------------------------
  Hℋᵣ(F, N) = H(F) * ℋᵣ(N)
  Ψtorq(F, ℋ₀, N) = (μ * (1 + χe) / J(F)) * (Hℋ₀(F, ℋ₀) ⋅ Hℋᵣ(F, N))
  ∂Ψtorq_∂H(F, ℋ₀, N) = (μ * (1 + χe) / J(F)) * (Hℋᵣ(F, N) ⊗ ℋ₀ + Hℋ₀(F, ℋ₀) ⊗ ℋᵣ(N))
  ∂Ψtorq_∂J(F, ℋ₀, N) = -(μ * (1 + χe) / J(F)^2) * (Hℋ₀(F, ℋ₀) ⋅ Hℋᵣ(F, N))
  ∂Ψtorq_∂u(F, ℋ₀, N) = (tr(∂Ψtorq_∂H(F, ℋ₀, N)) * I2) - ∂Ψtorq_∂H(F, ℋ₀, N)' + ∂Ψtorq_∂J(F, ℋ₀, N) * H(F)
  ∂Ψtorq_∂φ(F, ℋ₀, N) = (μ * (1 + χe) / J(F)) * (H(F)' * Hℋᵣ(F, N))

  ∂Ψtorq_∂HH(F, ℋ₀, N) = (μ * (1 + χe) / J(F)) * (I2 ⊗₁₃²⁴ (ℋᵣ(N) ⊗ ℋ₀ + ℋ₀ ⊗ ℋᵣ(N)))
  ∂Ψtorq_∂HJ(F, ℋ₀, N) = -(μ * (1 + χe) / J(F)^2) * (Hℋᵣ(F, N) ⊗ ℋ₀ + Hℋ₀(F, ℋ₀) ⊗ ℋᵣ(N))
  ∂Ψtorq_∂JJ(F, ℋ₀, N) = (μ * (1 + χe) / J(F)^3) * (Hℋ₀(F, ℋ₀) ⋅ Hℋᵣ(F, N))

  ∂Ψtorq_∂uu(F, ℋ₀, N) = _∂H∂F_2D()' * ∂Ψtorq_∂HH(F, ℋ₀, N) * _∂H∂F_2D() +
                         _∂H∂F_2D()' * (∂Ψtorq_∂HJ(F, ℋ₀, N) ⊗ H(F)) +
                         (H(F) ⊗ ∂Ψtorq_∂HJ(F, ℋ₀, N)) * _∂H∂F_2D() +
                         ∂Ψtorq_∂JJ(F, ℋ₀, N) * (H(F) ⊗ H(F)) +
                         ∂Ψtorq_∂J(F, ℋ₀, N) * _∂H∂F_2D()

  ∂Ψtorq_∂ℋ₀H(F, ℋ₀, N) = (μ / (J(F))) * ((I2 ⊗₁₃² Hℋᵣ(F, N)) + (H(F)' ⊗₁₂³ Hℋᵣ(F, N))) * (1 + χe)
  ∂Ψtorq_∂ℋ₀J(F, ℋ₀, N) = (-μ / (J(F))^2.0) * (H(F)' * Hℋᵣ(F, N)) * (1 + χe)


  ∂Ψtorq_∂φu(F, ℋ₀, N) = (∂Ψtorq_∂ℋ₀H(F, ℋ₀, N) * _∂H∂F_2D()) + (∂Ψtorq_∂ℋ₀J(F, ℋ₀, N) ⊗₁²³ H(F))


  #-------------------------------------------------------------------------------------
  #                           TOTAL ENERGY
  #-------------------------------------------------------------------------------------
  Ψ(F, ℋ₀, N) = Ψmm(F, ℋ₀) + βcoup * Ψcoup(F, N) + βmok * Ψmok(F, N) + Ψtorq(F, ℋ₀, N)
  ∂Ψ_u(F, ℋ₀, N) = ∂Ψmm_∂u(F, ℋ₀) + βcoup * ∂Ψcoup_∂u(F, N) + βmok * ∂Ψmok_∂u(F, N) + ∂Ψtorq_∂u(F, ℋ₀, N)
  ∂Ψ_φ(F, ℋ₀, N) = ∂Ψmm_∂φ(F, ℋ₀) + ∂Ψtorq_∂φ(F, ℋ₀, N)
  ∂Ψ_uu(F, ℋ₀, N) = ∂Ψmm_∂uu(F, ℋ₀) + βcoup * ∂Ψcoup_∂uu(F, N) + βmok * ∂Ψmok_∂uu(F, N) + ∂Ψtorq_∂uu(F, ℋ₀, N)
  ∂Ψ_φu(F, ℋ₀, N) = ∂Ψmm_∂φu(F, ℋ₀) + ∂Ψtorq_∂φu(F, ℋ₀, N)
  ∂Ψ_φφ(F, ℋ₀, N) = ∂Ψmm_∂φφ(F, ℋ₀)

  return (Ψ, ∂Ψ_u, ∂Ψ_φ, ∂Ψ_uu, ∂Ψ_φu, ∂Ψ_φφ)

end


end