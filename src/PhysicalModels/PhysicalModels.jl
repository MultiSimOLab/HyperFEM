module PhysicalModels

using Gridap
using Gridap.Helpers
using DrWatson

using ForwardDiff
using LinearAlgebra
using ..TensorAlgebra
using ..TensorAlgebra: _δδ_μ_3D
using ..TensorAlgebra: _δδ_λ_3D
using ..TensorAlgebra: _δδ_μ_2D
using ..TensorAlgebra: _δδ_λ_2D
using ..TensorAlgebra: _∂H∂F_2D
using ..TensorAlgebra: I3
using ..TensorAlgebra: I9
using StaticArrays


export NeoHookean3D
export IncompressibleNeoHookean3D
export IncompressibleNeoHookean2D
export IncompressibleNeoHookean2D_CV
export IncompressibleNeoHookean3D_2dP
export ARAP2D
export ARAP2D_regularized
export MoneyRivlin3D
export MoneyRivlin2D
export NonlinearMoneyRivlin3D
export NonlinearMoneyRivlin2D
export NonlinearMoneyRivlin2D_CV
export NonlinearNeoHookean_CV
export NonlinearIncompressibleMoneyRivlin2D_CV
export TransverseIsotropy3D
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
export MagnetoMechModel
export MagnetoVacuumModel

export Mechano
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

export DerivativeStrategy

export update_state!

struct DerivativeStrategy{Kind} end

abstract type PhysicalModel end
abstract type Mechano <: PhysicalModel end
abstract type Electro <: PhysicalModel end
abstract type Magneto <: PhysicalModel end
abstract type Thermo <: PhysicalModel end

abstract type MultiPhysicalModel <: PhysicalModel end
abstract type ElectroMechano <: MultiPhysicalModel end
abstract type ThermoElectroMechano <: MultiPhysicalModel end
abstract type ThermoMechano <: MultiPhysicalModel end
abstract type ThermoElectro <: MultiPhysicalModel end
abstract type FlexoElectro <: MultiPhysicalModel end
abstract type MagnetoMechano <: MultiPhysicalModel end

include("KinematicModels.jl")
include("PINNs.jl")
export Kinematics
export EvolutiveKinematics
export get_Kinematics
export getIsoInvariants

export HessianRegularization
export Hessian∇JRegularization

# ============================================
# Regularization of Mechanical models
# ============================================

struct HessianRegularization{A,B} <: Mechano
  Mechano::A
  δ::Float64
  Kinematic::B
  function HessianRegularization(; Mechano::Mechano, δ::Float64=1.0e-6)
    new{typeof(Mechano),typeof(Mechano.Kinematic)}(Mechano, δ, Mechano.Kinematic)
  end


  function (obj::HessianRegularization)(Λ::Float64=1.0)
    Ψs, ∂Ψs, ∂2Ψs = obj.Mechano()
    δ = obj.δ

    ∂2Ψ(F) = begin
      vecval = eigen(get_array(∂2Ψs(F)))
      vec = real(vecval.vectors)
      val = real(vecval.values)
      TensorValue(vec * diagm(max.(δ, val)) * vec')
    end

    return (Ψs, ∂Ψs, ∂2Ψ)
  end



end


struct Hessian∇JRegularization{A,B} <: Mechano
  Mechano::A
  δ::Float64
  κ::Float64
  Kinematic::B
  function Hessian∇JRegularization(; Mechano::Mechano, δ::Float64=1.0e-6, κ::Float64=1.0)
    new{typeof(Mechano),typeof(Mechano.Kinematic)}(Mechano, δ, κ, Mechano.Kinematic)
  end


  function (obj::Hessian∇JRegularization)(Λ::Float64=1.0)
    Ψs, ∂Ψs, ∂2Ψs = obj.Mechano()
    _, H, J = get_Kinematics(obj.Mechano.Kinematic; Λ=Λ)
    δ, κ = obj.δ, obj.κ

    Ψ(F, Jh) = Ψs(F) + 0.5 * κ * (J(F) - Jh)^2
    ∂Ψ(F, Jh) = ∂Ψs(F) + κ * (J(F) - Jh) * H(F)
    ∂2Ψ_(F, Jh) = ∂2Ψs(F) + κ * (H(F) ⊗ H(F)) + κ * (J(F) - Jh) * _∂H∂F_2D()

    ∂2Ψ(F,Jh) = begin
      vecval = eigen(get_array(∂2Ψ_(F, Jh)))
      vec = real(vecval.vectors)
      val = real(vecval.values)
      TensorValue(vec * diagm(max.(δ, val)) * vec')
    end

    # ∂2Ψ(F, Jh) = TensorValue(real(λ(F, Jh).vectors) * diagm(max.(δ, real(λ(F, Jh).values))) * real(λ(F, Jh).vectors)')
    return (Ψ, ∂Ψ, ∂2Ψ)
  end
end


# ======================
# Energy interpolations
# ======================
struct EnergyInterpolationScheme{A,B} <: PhysicalModel
  p::Float64
  model1::A
  model2::B
  function EnergyInterpolationScheme(model1, model2; p::Float64=3.0)
    new{typeof(model1),typeof(model2)}(p, model1, model2)
  end

  function (obj::EnergyInterpolationScheme{<:Mechano,<:Mechano})()
    Ψs, ∂Ψs, ∂2Ψs = obj.model1()
    Ψv, ∂Ψv, ∂2Ψv = obj.model2()
    p = obj.p

    Ψ(ρ, F) = ρ^p * Ψs(F) + (1 - ρ^p) * Ψv(F)
    DΨ_Dρ(ρ, F) = p * ρ^(p - 1) * Ψs(F) - (p * ρ^(p - 1)) * Ψv(F)

    ∂Ψ(ρ, F) = ρ^p * ∂Ψs(F) + (1 - ρ^p) * ∂Ψv(F)
    D∂Ψ_Dρ(ρ, F) = p * ρ^(p - 1) * ∂Ψs(F) - (p * ρ^(p - 1)) * ∂Ψv(F)

    ∂2Ψ(ρ, F) = ρ^p * ∂2Ψs(F) + (1 - ρ^p) * ∂2Ψv(F)
    D∂2Ψ_Dρ(ρ, F) = p * ρ^(p - 1) * ∂2Ψs(F) - (p * ρ^(p - 1)) * ∂2Ψv(F)

    return (Ψ, ∂Ψ, ∂2Ψ, DΨ_Dρ, D∂Ψ_Dρ, D∂2Ψ_Dρ)
  end
end



# ===================
# Magneto models
# ===================

struct IdealMagnetic{A} <: Magneto
  μ::Float64
  χe::Float64
  Kinematic::A
  function IdealMagnetic(; μ::Float64, χe::Float64=0.0, Kinematic::KinematicModel=Kinematics(Magneto))
    new{typeof(Kinematic)}(μ, χe, Kinematic)
  end
end

struct IdealMagnetic2D{A} <: Magneto
  μ::Float64
  χe::Float64
  Kinematic::A
  function IdealMagnetic2D(; μ::Float64, χe::Float64=0.0, Kinematic::KinematicModel=Kinematics(Magneto))
    new{typeof(Kinematic)}(μ, χe, Kinematic)
  end
end

struct HardMagnetic{A} <: Magneto
  μ::Float64
  αr::Float64
  χe::Float64
  χr::Float64
  βmok::Float64
  βcoup::Float64
  Kinematic::A
  function HardMagnetic(; μ::Float64, αr::Float64, χe::Float64=0.0, χr::Float64=8.0, βmok::Float64=1.0, βcoup::Float64=1.0, Kinematic::KinematicModel=Kinematics(Magneto))
    new{typeof(Kinematic)}(μ, αr, χe, χr, βmok, βcoup, Kinematic)
  end
end


struct HardMagnetic2D{A} <: Magneto
  μ::Float64
  αr::Float64
  χe::Float64
  χr::Float64
  βmok::Float64
  βcoup::Float64
  Kinematic::A
  function HardMagnetic2D(; μ::Float64, αr::Float64, χe::Float64=0.0, χr::Float64=8.0, βmok::Float64=1.0, βcoup::Float64=1.0, Kinematic::KinematicModel=Kinematics(Magneto))
    new{typeof(Kinematic)}(μ, αr, χe, χr, βmok, βcoup, Kinematic)
  end
end


# ===================
# Electro models
# ===================

struct IdealDielectric{A} <: Electro
  ε::Float64
  Kinematic::A
  function IdealDielectric(; ε::Float64, Kinematic::KinematicModel=Kinematics(Electro))
    new{typeof(Kinematic)}(ε, Kinematic)
  end
end

# ===================
# Thermal models
# ===================

struct ThermalModel <: Thermo
  Cv::Float64
  θr::Float64
  α::Float64
  κ::Float64
  function ThermalModel(; Cv::Float64, θr::Float64, α::Float64, κ::Float64=10.0)
    new(Cv, θr, α, κ)
  end

  function (obj::ThermalModel)(Λ::Float64=1.0)
    Ψ(δθ) = obj.Cv * (δθ - (δθ + obj.θr) * log((δθ + obj.θr) / obj.θr))
    ∂Ψθ(δθ) = -obj.Cv * log((δθ + obj.θr) / obj.θr)
    ∂Ψθθ(δθ) = -obj.Cv / (δθ + obj.θr)
    return (Ψ, ∂Ψθ, ∂Ψθθ)
  end

end


# ===================
# Mechanical models
# ===================

struct LinearElasticity2D{A} <: Mechano
  λ::Float64
  μ::Float64
  ρ::Float64
  Kinematic::A
  function LinearElasticity2D(; λ::Float64, μ::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ, ρ, Kinematic)
  end

  function (obj::LinearElasticity2D)(Λ::Float64=1.0)
    λ, μ = obj.λ, obj.μ
    I22 = I2()
    ∂Ψuu(F) = _δδ_μ_2D(μ) + _δδ_λ_2D(λ)
    ∂Ψu(F) = ∂Ψuu(F) ⊙ (F - I22)
    Ψ(F) = 0.5 * (F - I22) ⊙ (∂Ψuu(F) ⊙ (F - I22))
    return (Ψ, ∂Ψu, ∂Ψuu)
  end


end

mutable struct LinearElasticity3D{A} <: Mechano
  λ::Float64
  μ::Float64
  ρ::Float64
  Kinematic::A
  function LinearElasticity3D(; λ::Float64, μ::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ, ρ, Kinematic)
  end

  function (obj::LinearElasticity3D)(Λ::Float64=1.0)
    λ, μ = obj.λ, obj.μ
    I33 = I3()
    ∂Ψuu(F) = _δδ_μ_3D(μ) + _δδ_λ_3D(λ)
    ∂Ψu(F) = ∂Ψuu(F) ⊙ (F - I33)
    Ψ(F) = 0.5 * (F - I33) ⊙ (∂Ψuu(F) ⊙ (F - I33))
    return (Ψ, ∂Ψu, ∂Ψuu)
  end


end

struct NeoHookean3D{A} <: Mechano
  λ::Float64
  μ::Float64
  ρ::Float64
  Kinematic::A
  function NeoHookean3D(; λ::Float64, μ::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ, ρ, Kinematic)
  end

  function (obj::NeoHookean3D)(Λ::Float64=1.0; Threshold=0.01)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    λ, μ = obj.λ, obj.μ
    Ψ(F) = μ / 2 * tr((F)' * F) - μ * logreg(J(F)) + (λ / 2) * (J(F) - 1)^2 - 3.0 * (μ / 2.0)

    ∂log∂J(J) = J >= Threshold ? 1 / J : (2 / Threshold - J / (Threshold^2))
    ∂log2∂J2(J) = J >= Threshold ? -1 / (J^2) : (-1 / (Threshold^2))

    ∂Ψ_∂J(F) = -μ * ∂log∂J(J(F)) + λ * (J(F) - 1)
    ∂Ψu(F) = μ * F + ∂Ψ_∂J(F) * H(F)
    I_ = I9()
    ∂Ψ2_∂J2(F) = -μ * ∂log2∂J2(J(F)) + λ
    ∂Ψuu(F) = μ * I_ + ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂J(F) * ×ᵢ⁴(F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end

end

struct MoneyRivlin3D{A} <: Mechano
  λ::Float64
  μ1::Float64
  μ2::Float64
  ρ::Float64
  Kinematic::A
  function MoneyRivlin3D(; λ::Float64, μ1::Float64, μ2::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ1, μ2, ρ, Kinematic)
  end



  function (obj::MoneyRivlin3D)(Λ::Float64=1.0; Threshold=0.01)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    λ, μ1, μ2 = obj.λ, obj.μ1, obj.μ2
    Ψ(F) = μ1 / 2 * tr((F)' * F) + μ2 / 2.0 * tr((H(F))' * H(F)) - (μ1 + 2 * μ2) * logreg(J(F)) +
           (λ / 2.0) * (J(F) - 1)^2 - (3.0 / 2.0) * (μ1 + μ2)
    ∂Ψ_∂F(F) = μ1 * F
    ∂Ψ_∂H(F) = μ2 * H(F)
    ∂log∂J(J) = J >= Threshold ? 1 / J : (2 / Threshold - J / (Threshold^2))
    ∂log2∂J2(J) = J >= Threshold ? -1 / (J^2) : (-1 / (Threshold^2))
    ∂Ψ_∂J(F) = -(μ1 + 2.0 * μ2) * ∂log∂J(J(F)) + λ * (J(F) - 1)
    ∂Ψ2_∂J2(F) = -(μ1 + 2.0 * μ2) * ∂log2∂J2(J(F)) + λ
    # ∂Ψ_∂J(F) = -(μ1 + 2.0 * μ2) / J(F) + λ * (J(F) - 1)
    # ∂Ψ2_∂J2(F) = (μ1 + 2.0 * μ2) / (J(F)^2) + λ

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂H(F) × F + ∂Ψ_∂J(F) * H(F)
    # I_ = TensorValue(Matrix(1.0I, 9, 9))
    I_ = I9()
    # ∂Ψuu(∇u) = μ1 * I_ + μ2 * (F × (I_ × F)) + ∂Ψ2_∂J2(∇u) * (H(F) ⊗ H(F)) + (I_ × (∂Ψ_∂H(∇u) + ∂Ψ_∂J(∇u) * F))
    ∂Ψuu(F) = μ1 * I_ + μ2 * (F × (I_ × F)) + ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ×ᵢ⁴(∂Ψ_∂H(F) + ∂Ψ_∂J(F) * F)

    return (Ψ, ∂Ψu, ∂Ψuu)

  end


end

struct MoneyRivlin2D{A} <: Mechano
  λ::Float64
  μ1::Float64
  μ2::Float64
  ρ::Float64
  Kinematic::A

  function MoneyRivlin2D(; λ::Float64, μ1::Float64, μ2::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ1, μ2, ρ, Kinematic)
  end

  function (obj::MoneyRivlin2D)(Λ::Float64=1.0; Threshold=0.01)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    λ, μ1, μ2 = obj.λ, obj.μ1, obj.μ2
    Ψ(F) = (μ1 / 2 + μ2 / 2) * tr((F)' * F) + μ2 / 2.0 * J(F)^2 - (μ1 + 2 * μ2) * logreg(J(F)) +
           (λ / 2.0) * (J(F) - 1)^2
    ∂Ψ_(F) = ForwardDiff.gradient(F -> Ψ(F), get_array(F))
    ∂2Ψ_(F) = ForwardDiff.jacobian(F -> ∂Ψ_(F), get_array(F))

    ∂Ψu(F) = TensorValue(∂Ψ_(F))
    ∂Ψuu(F) = TensorValue(∂2Ψ_(F))

    return (Ψ, ∂Ψu, ∂Ψuu)

  end


end


struct NonlinearMoneyRivlin3D{A} <: Mechano
  λ::Float64
  μ1::Float64
  μ2::Float64
  α::Float64
  β::Float64
  ρ::Float64
  Kinematic::A
  function NonlinearMoneyRivlin3D(; λ::Float64, μ1::Float64, μ2::Float64, α::Float64, β::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ1, μ2, α, β, ρ, Kinematic)
  end

  function (obj::NonlinearMoneyRivlin3D)(Λ::Float64=1.0; Threshold=0.01)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    λ, μ1, μ2, α, β = obj.λ, obj.μ1, obj.μ2, obj.α, obj.β
    Ψ(F) = μ1 / (2.0 * α * 3.0^(α - 1)) * (tr((F)' * F))^α + μ2 / (2.0 * β * 3.0^(β - 1)) * (tr((H(F))' * H(F)))^β - (μ1 + 2 * μ2) * logreg(J(F)) +
           (λ / 2.0) * (J(F) - 1)^2

    ∂Ψ_∂F(F) = (μ1 / (3.0^(α - 1)) * (tr((F)' * F))^(α - 1)) * F
    ∂Ψ_∂H(F) = (μ2 / (3.0^(β - 1)) * (tr((H(F))' * H(F)))^(β - 1)) * H(F)
    ∂log∂J(J) = J >= Threshold ? 1 / J : (2 / Threshold - J / (Threshold^2))
    ∂log2∂J2(J) = J >= Threshold ? -1 / (J^2) : (-1 / (Threshold^2))
    ∂Ψ_∂J(F) = -(μ1 + 2.0 * μ2) * ∂log∂J(J(F)) + λ * (J(F) - 1)
    ∂Ψ2_∂J2(F) = -(μ1 + 2.0 * μ2) * ∂log2∂J2(J(F)) + λ

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂H(F) × F + ∂Ψ_∂J(F) * H(F)
    I_ = I9()
    ∂ΨFF(F) = (2 * μ1 * (α - 1) / (3.0^(α - 1)) * (tr((F)' * F))^(α - 2)) * (F ⊗ F) + (μ1 / (3.0^(α - 1)) * (tr((F)' * F))^(α - 1)) * I_
    ∂ΨHH(F) = (2 * μ2 * (β - 1) / (3.0^(β - 1)) * (tr((H(F))' * H(F)))^(β - 2)) * (H(F) ⊗ H(F)) + (μ2 / (3.0^(β - 1)) * (tr((H(F))' * H(F)))^(β - 1)) * I_
    ∂Ψuu(F) = ∂ΨFF(F) + (F × (∂ΨHH(F) × F)) + ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ×ᵢ⁴(∂Ψ_∂H(F) + ∂Ψ_∂J(F) * F)

    return (Ψ, ∂Ψu, ∂Ψuu)

  end


end


struct NonlinearMoneyRivlin2D{A} <: Mechano
  λ::Float64
  μ1::Float64
  μ2::Float64
  α::Float64
  β::Float64
  ρ::Float64
  Kinematic::A
  function NonlinearMoneyRivlin2D(; λ::Float64, μ1::Float64, μ2::Float64, α::Float64, β::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ1, μ2, α, β, ρ, Kinematic)
  end

  function (obj::NonlinearMoneyRivlin2D)(Λ::Float64=1.0; Threshold=0.01)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    λ, μ1, μ2, α, β = obj.λ, obj.μ1, obj.μ2, obj.α, obj.β

    Ψ(F) = μ1 / (2.0 * α * 3.0^(α - 1)) * (tr((F)' * F) + 1.0)^α + μ2 / (2.0 * β * 3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^β - (μ1 + 2.0 * μ2) * logreg(J(F)) +
           (λ / 2.0) * (J(F) - 1)^2

    ∂Ψ_∂F(F) = ((μ1 / (3.0^(α - 1)) * (tr((F)' * F) + 1.0)^(α - 1)) + μ2 / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 1)) * F
    ∂log∂J(J) = J >= Threshold ? 1 / J : (2 / Threshold - J / (Threshold^2))
    ∂log2∂J2(J) = J >= Threshold ? -1 / (J^2) : (-1 / (Threshold^2))
    ∂Ψ_∂J(F) = μ2 / (3.0^(β - 1)) * J(F) * (tr((F)' * F) + J(F)^2)^(β - 1) - (μ1 + 2.0 * μ2) * ∂log∂J(J(F)) + λ * (J(F) - 1)

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)
    I_ = I4()

    ∂Ψ2_∂FF(F) = ((μ1 / (3.0^(α - 1)) * (tr((F)' * F) + 1.0)^(α - 1)) + μ2 / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 1)) * I_ +
                 2 * ((μ1 * (α - 1) / (3.0^(α - 1)) * (tr((F)' * F) + 1.0)^(α - 2)) + μ2 * (β - 1) / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 2)) * (F ⊗ F)
    ∂Ψ2_∂FJ(F) = (2 * μ2 * (β - 1) / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 2)) * J(F) * F
    ∂Ψ2_∂JJ(F) = μ2 / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 1) + (2 * μ2 * (β - 1) / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 2)) * J(F)^2 - (μ1 + 2.0 * μ2) * ∂log2∂J2(J(F)) + λ


    ∂Ψuu(F) = ∂Ψ2_∂FF(F) + (∂Ψ2_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ2_∂FJ(F)) + ∂Ψ2_∂JJ(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂J(F) * _∂H∂F_2D()

    return (Ψ, ∂Ψu, ∂Ψuu)

  end

end



struct NonlinearMoneyRivlin2D_CV{A} <: Mechano
  λ::Float64
  μ1::Float64
  μ2::Float64
  α::Float64
  β::Float64
  γ::Float64
  ρ::Float64
  Kinematic::A
  function NonlinearMoneyRivlin2D_CV(; λ::Float64, μ1::Float64, μ2::Float64, α::Float64, β::Float64, γ::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ1, μ2, α, β, γ, ρ, Kinematic)
  end

  function (obj::NonlinearMoneyRivlin2D_CV)(Λ::Float64=1.0)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    λ, μ1, μ2, α, β, γ = obj.λ, obj.μ1, obj.μ2, obj.α, obj.β, obj.γ

    Ψ(F) = μ1 / (2.0 * α * 3.0^(α - 1)) * (tr((F)' * F) + 1.0)^α + μ2 / (2.0 * β * 3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^β - (μ1 + 2.0 * μ2) * log(J(F)) +
           (λ) * (J(F)^(γ) + J(F)^(-γ))

    ∂Ψ_∂F(F) = ((μ1 / (3.0^(α - 1)) * (tr((F)' * F) + 1.0)^(α - 1)) + μ2 / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 1)) * F
    ∂Ψ_∂J(F) = μ2 / (3.0^(β - 1)) * J(F) * (tr((F)' * F) + J(F)^2)^(β - 1) - (μ1 + 2.0 * μ2) * (1.0 / J(F)) + λ * γ * (J(F)^(γ - 1) - J(F)^(-γ - 1))

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)
    I_ = I4()

    ∂Ψ2_∂FF(F) = ((μ1 / (3.0^(α - 1)) * (tr((F)' * F) + 1.0)^(α - 1)) + μ2 / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 1)) * I_ +
                 2 * ((μ1 * (α - 1) / (3.0^(α - 1)) * (tr((F)' * F) + 1.0)^(α - 2)) + μ2 * (β - 1) / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 2)) * (F ⊗ F)
    ∂Ψ2_∂FJ(F) = (2 * μ2 * (β - 1) / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 2)) * J(F) * F
    ∂Ψ2_∂JJ(F) = μ2 / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 1) + (2 * μ2 * (β - 1) / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 2)) * J(F)^2 + (μ1 + 2.0 * μ2) * (1.0 / (J(F))^2) + λ * γ * ((γ - 1) * J(F)^(γ - 2) + (γ + 1) * J(F)^(-γ - 2))

    ∂Ψuu(F) = ∂Ψ2_∂FF(F) + (∂Ψ2_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ2_∂FJ(F)) + ∂Ψ2_∂JJ(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂J(F) * _∂H∂F_2D()

    return (Ψ, ∂Ψu, ∂Ψuu)

  end


end





struct NonlinearNeoHookean_CV{A} <: Mechano
  λ::Float64
  μ::Float64
  α::Float64
  γ::Float64
  ρ::Float64
  Kinematic::A
  function NonlinearNeoHookean_CV(; λ::Float64, μ::Float64, α::Float64, γ::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ, α, γ, ρ, Kinematic)
  end

  function (obj::NonlinearNeoHookean_CV)(Λ::Float64=1.0)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    λ, μ, α, γ = obj.λ, obj.μ, obj.α,  obj.γ

    Ψ(F) = μ / (2.0 * α * 3.0^(α - 1)) * (tr((F)' * F) + 1.0)^α - μ  * log(J(F)) + λ * (J(F)^(γ) + J(F)^(-γ))

    ∂Ψ_∂F(F) = ((μ / (3.0^(α - 1)) * (tr((F)' * F) + 1.0)^(α - 1))) * F
    ∂Ψ_∂J(F) = - μ * (1.0 / J(F)) + λ * γ * (J(F)^(γ - 1) - J(F)^(-γ - 1))

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)
    I_ = I9()

    ∂Ψ2_∂FF(F) = ((μ / (3.0^(α - 1)) * (tr((F)' * F) + 1.0)^(α - 1))) * I_ +
                 2 * ((μ * (α - 1) / (3.0^(α - 1)) * (tr((F)' * F) + 1.0)^(α - 2)) ) * (F ⊗ F)
    ∂Ψ2_∂JJ(F) = μ * (1.0 / (J(F))^2) + λ * γ * ((γ - 1) * J(F)^(γ - 2) + (γ + 1) * J(F)^(-γ - 2))

    ∂Ψuu(F) = ∂Ψ2_∂FF(F) + ∂Ψ2_∂JJ(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂J(F) * ×ᵢ⁴(F)

    return (Ψ, ∂Ψu, ∂Ψuu)

  end


end


struct NonlinearIncompressibleMoneyRivlin2D_CV{A} <: Mechano
  λ::Float64
  μ::Float64
  α::Float64
  γ::Float64
  ρ::Float64
  Kinematic::A
  function NonlinearIncompressibleMoneyRivlin2D_CV(; λ::Float64, μ::Float64, α::Float64, γ::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ, α, γ, ρ, Kinematic)
  end


  function (obj::NonlinearIncompressibleMoneyRivlin2D_CV)(Λ::Float64=1.0)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    λ, μ, α, γ = obj.λ, obj.μ, obj.α, obj.γ

    I_           =  I4()
    e(F)         =  (tr((F)' * F) + 1.0) * J(F)^(-2 / 3)
    ∂e_∂F(F)     =  2 * J(F)^(-2 / 3) * F
    ∂e_∂J(F)     =  -(2 / 3) * (tr((F)' * F) + 1.0) * J(F)^(-5 / 3)    
    ∂e2_∂F2(F)   =  2 * J(F)^(-2 / 3) * I_
    ∂e2_∂J2(F)   =  (10 / 9)*J(F)^(-8 / 3) * (tr((F)' * F) + 1.0)
    ∂e2_∂FJ(F)   =  -(4 / 3)*J(F)^(-5 / 3) * F

    Ψ1(F)        =  μ / (2* α) * (e(F))^α
    Ψ2(F)        =  (λ ) * (J(F)^(γ) + J(F)^(-γ))
    Ψ(F)         =  Ψ1(F) + Ψ2(F) 

    ∂Ψ1_∂F(F)    =  (μ / 2) * (((e(F))^(α - 1.0)) * ∂e_∂F(F))
    ∂Ψ1_∂J(F)    =  (μ / 2) * (((e(F))^(α - 1.0)) * ∂e_∂J(F))
    ∂Ψ2_∂J(F)    =  λ * γ *  (J(F)^(γ-1) - J(F)^(-γ-1))    
    ∂Ψ_∂F(F)     =  ∂Ψ1_∂F(F)
    ∂Ψ_∂J(F)     =  ∂Ψ1_∂J(F) + ∂Ψ2_∂J(F)
    ∂Ψu(F)       =  ∂Ψ_∂F(F)  + ∂Ψ_∂J(F) * H(F)

    ∂Ψ1_∂F2(F)   =  (μ / 2) * ((e(F)^(α - 1)) * ∂e2_∂F2(F) + (α - 1) * (e(F)^(α - 2)) * ∂e_∂F(F) ⊗ ∂e_∂F(F))
    ∂Ψ1_∂J2(F)   =  (μ / 2) * ((e(F)^(α - 1)) * ∂e2_∂J2(F) + (α - 1) * (e(F)^(α - 2)) * ∂e_∂J(F) * ∂e_∂J(F))
    ∂Ψ1_∂FJ(F)   =  (μ / 2) * ((e(F)^(α - 1)) * ∂e2_∂FJ(F) + (α - 1) * (e(F)^(α - 2)) * ∂e_∂F(F) * ∂e_∂J(F))
    ∂Ψ2_∂J2(F)   =  λ * γ  *  ((γ-1)*J(F)^(γ-2) + (γ+1)*J(F)^(-γ-2))

    ∂Ψ_∂F2(F)    =  ∂Ψ1_∂F2(F)
    ∂Ψ_∂FJ(F)    =  ∂Ψ1_∂FJ(F)
    ∂Ψ_∂J2(F)    =  ∂Ψ1_∂J2(F)  + ∂Ψ2_∂J2(F)

    ∂Ψuu(F)      =  ∂Ψ_∂F2(F) + ∂Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ_∂FJ(F) + ∂Ψ_∂J(F) * _∂H∂F_2D()
    
    return (Ψ, ∂Ψu, ∂Ψuu)
  end

end

struct TransverseIsotropy3D{A} <: Mechano
  μ::Float64
  α::Float64
  β::Float64
  ρ::Float64
  Kinematic::A
  function TransverseIsotropy3D(; μ::Float64, α::Float64, β::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(μ, α, β, ρ, Kinematic)
  end


  function (obj::TransverseIsotropy3D)(Λ::Float64=1.0; Threshold=0.01)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    I4(F, N) = (F * N) ⋅ (F * N)
    I5(F, N) = (H(F) * N) ⋅ (H(F) * N)
    μ, α, β = obj.μ, obj.α, obj.β
    Ψ(F, N) = μ / (2.0 * α) * (I4(F, N)^α - 1) + μ / (2.0 * β) * (I5(F, N)^β - 1) - μ * logreg(J(F))

    ∂Ψ_∂F(F, N) = (μ * (I4(F, N)^(α - 1))) * ((F * N) ⊗ N)
    ∂Ψ_∂H(F, N) = (μ * (I5(F, N)^(β - 1))) * ((H(F) * N) ⊗ N)
    ∂log∂J(J) = J >= Threshold ? 1 / J : (2 / Threshold - J / (Threshold^2))
    ∂log2∂J2(J) = J >= Threshold ? -1 / (J^2) : (-1 / (Threshold^2))
    ∂Ψ_∂J(F, N) = -μ * ∂log∂J(J(F))
    ∂Ψ2_∂J2(F, N) = -μ * ∂log2∂J2(J(F))

    ∂Ψu(F, N) = ∂Ψ_∂F(F, N) + ∂Ψ_∂H(F, N) × F + ∂Ψ_∂J(F, N) * H(F)
    I__ = I3()
    ∂ΨFF(F, N) = μ * (I4(F, N)^(α - 1)) * (I__ ⊗₁₃²⁴ (N ⊗ N)) + 2μ * (α - 1) * I4(F, N)^(α - 2) * (((F * N) ⊗ N) ⊗ ((F * N) ⊗ N))
    ∂ΨHH(F, N) = μ * (I5(F, N)^(β - 1)) * (I__ ⊗₁₃²⁴ (N ⊗ N)) + 2μ * (β - 1) * I5(F, N)^(β - 2) * (((H(F) * N) ⊗ N) ⊗ ((H(F) * N) ⊗ N))
    ∂Ψuu(F, N) = ∂ΨFF(F, N) + (F × (∂ΨHH(F, N) × F)) + ∂Ψ2_∂J2(F, N) * (H(F) ⊗ H(F)) + ×ᵢ⁴(∂Ψ_∂H(F, N) + ∂Ψ_∂J(F, N) * F)

    return (Ψ, ∂Ψu, ∂Ψuu)

  end


end


struct IncompressibleNeoHookean3D{A} <: Mechano
  λ::Float64
  μ::Float64
  ρ::Float64
  δ::Float64
  Kinematic::A
  function IncompressibleNeoHookean3D(; λ::Float64, μ::Float64, ρ::Float64=0.0, δ::Float64=0.1, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ, ρ, δ, Kinematic)
  end

  function (obj::IncompressibleNeoHookean3D)(Λ::Float64=1.0)
    _, H, J_ = get_Kinematics(obj.Kinematic; Λ=Λ)
    λ, μ, δ = obj.λ, obj.μ, obj.δ
    J(F) = 0.5 * (J_(F) + sqrt(J_(F)^2 + δ^2))
    ∂J(F) = 0.5 * (1.0 + J_(F) / sqrt(J_(F)^2 + δ^2))
    ∂2J(F) = 0.5 * δ^2 / ((J_(F)^2 + δ^2)^(3 / 2))


    J1 = 0.5 * (1.0 + sqrt(1.0 + δ^2))
    ∂J1 = 0.5 * (1.0 + 1.0 / sqrt(1.0^2 + δ^2))
    β = μ * (J1^(-2 / 3) - J1^(-5 / 3) * ∂J1)
    Ψ1(F) = μ / 2 * (tr((F)' * F)) * J(F)^(-2 / 3)
    Ψ2(F) = (λ / 2) * (J_(F) - 1)^2
    Ψ(F) = Ψ1(F) + Ψ2(F) - β * log(J_(F))


    ∂Ψ1_∂J(F) = -μ / 3 * (tr((F)' * F)) * J(F)^(-5 / 3)
    ∂Ψ2_∂J(F) = λ * (J_(F) - 1)
    ∂Ψ3_∂J(F) = -β / J_(F)
    ∂Ψ_∂J(F) = ∂Ψ1_∂J(F) * ∂J(F) + ∂Ψ2_∂J(F) + ∂Ψ3_∂J(F)


    ∂Ψu(F) = μ * F * J(F)^(-2 / 3) + (∂Ψ_∂J(F) * ∂J(F)) * H(F)
    I_ = I9()

    ∂Ψ1_∂J2(F) = (5 / 9) * μ * J(F)^(-8 / 3) * (tr((F)' * F))
    ∂Ψ2_∂J2(F) = λ
    ∂Ψ3_∂J2(F) = β / J_(F)^2
    ∂Ψ_∂J2(F) = (∂Ψ1_∂J2(F) * ∂J(F)^2 + ∂Ψ1_∂J(F) * ∂2J(F)) + ∂Ψ2_∂J2(F) + ∂Ψ3_∂J2(F)
    ∂Ψ_∂FJ(F) = -(2 / 3) * μ * J(F)^(-5 / 3) * ∂J(F) * F

    ∂Ψuu(F) = μ * I_ * J(F)^(-2 / 3) + ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ_∂FJ(F) + ∂Ψ_∂J(F) * ×ᵢ⁴(F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end

end

struct IncompressibleNeoHookean2D{A} <: Mechano
  λ::Float64
  μ::Float64
  ρ::Float64
  δ::Float64
  Kinematic::A
  function IncompressibleNeoHookean2D(; λ::Float64, μ::Float64, ρ::Float64=0.0, δ::Float64=0.1, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ, ρ, δ, Kinematic)
  end

  function (obj::IncompressibleNeoHookean2D)(Λ::Float64=1.0)
    _, H, J_ = get_Kinematics(obj.Kinematic; Λ=Λ)
    λ, μ, δ = obj.λ, obj.μ, obj.δ

    J(F) = 0.5 * (J_(F) + sqrt(J_(F)^2 + δ^2))
    ∂J(F) = 0.5 * (1.0 + J_(F) / sqrt(J_(F)^2 + δ^2))
    ∂2J(F) = 0.5 * δ^2 / ((J_(F)^2 + δ^2)^(3 / 2))

    J1 = 0.5 * (1.0 + sqrt(1.0 + δ^2))
    ∂J1 = 0.5 * (1.0 + 1.0 / sqrt(1.0^2 + δ^2))
    β = μ * (J1^(-2 / 3) - J1^(-5 / 3) * ∂J1)
    Ψ1(F) = μ / 2 * (tr((F)' * F) + 1.0) * J(F)^(-2 / 3)
    Ψ2(F) = (λ / 2) * (J_(F) - 1)^2
    Ψ(F) = Ψ1(F) + Ψ2(F) - β * log(J_(F))

    ∂Ψ1_∂J(F) = -μ / 3 * (tr((F)' * F) + 1.0) * J(F)^(-5 / 3)
    ∂Ψ2_∂J(F) = λ * (J_(F) - 1)
    ∂Ψ3_∂J(F) = -β / J_(F)
    ∂Ψ_∂J(F) = ∂Ψ1_∂J(F) * ∂J(F) + ∂Ψ2_∂J(F) + ∂Ψ3_∂J(F)

    ∂Ψu(F) = μ * F * J(F)^(-2 / 3) + ∂Ψ_∂J(F) * H(F)
    I_ = I4()
    ∂Ψ1_∂J2(F) = (5 / 9) * μ * J(F)^(-8 / 3) * (tr((F)' * F) + 1.0)
    ∂Ψ2_∂J2(F) = λ
    ∂Ψ3_∂J2(F) = β / J_(F)^2
    ∂Ψ_∂J2(F) = (∂Ψ1_∂J2(F) * ∂J(F)^2 + ∂Ψ1_∂J(F) * ∂2J(F)) + ∂Ψ2_∂J2(F) + ∂Ψ3_∂J2(F)
    ∂Ψ_∂FJ(F) = -(2 / 3) * μ * J(F)^(-5 / 3) * ∂J(F) * F
    ∂Ψuu(F) = μ * I_ * J(F)^(-2 / 3) + ∂Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ_∂FJ(F) + ∂Ψ_∂J(F) * _∂H∂F_2D()
    return (Ψ, ∂Ψu, ∂Ψuu)
  end

end

struct IncompressibleNeoHookean2D_CV{A} <: Mechano
  λ::Float64
  μ::Float64
  γ::Float64
  ρ::Float64
  Kinematic::A
  function IncompressibleNeoHookean2D_CV(; λ::Float64, μ::Float64, γ::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ, γ, ρ, Kinematic)
  end

  function (obj::IncompressibleNeoHookean2D_CV)(Λ::Float64=1.0)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    λ, μ, γ = obj.λ, obj.μ, obj.γ

    Ψ1(F) = μ / 2 * (tr((F)' * F) + 1.0) * J(F)^(-2 / 3)
    Ψ2(F) = λ * (J(F)^(γ) + J(F)^(-γ))
    Ψ(F) = Ψ1(F) + Ψ2(F)

    ∂Ψ1_∂J(F) = -μ / 3 * (tr((F)' * F) + 1.0) * J(F)^(-5 / 3)
    ∂Ψ2_∂J(F) = λ * γ * (J(F)^(γ - 1) - J(F)^(-γ - 1))
    ∂Ψ_∂J(F) = ∂Ψ1_∂J(F) + ∂Ψ2_∂J(F)
    ∂Ψu(F) = μ * F * J(F)^(-2 / 3) + ∂Ψ_∂J(F) * H(F)

    I_ = I4()
    ∂Ψ1_∂J2(F) = (5 / 9) * μ * J(F)^(-8 / 3) * (tr((F)' * F) + 1.0)
    ∂Ψ2_∂J2(F) = λ * γ * ((γ - 1) * J(F)^(γ - 2) + (γ + 1) * J(F)^(-γ - 2))
    ∂Ψ_∂J2(F) = ∂Ψ1_∂J2(F) + ∂Ψ2_∂J2(F)
    ∂Ψ_∂FJ(F) = -(2 / 3) * μ * J(F)^(-5 / 3) * F
    ∂Ψuu(F) = μ * I_ * J(F)^(-2 / 3) + ∂Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ_∂FJ(F) + ∂Ψ_∂J(F) * _∂H∂F_2D()
    return (Ψ, ∂Ψu, ∂Ψuu)


  end

end




struct ARAP2D_regularized{A} <: Mechano
  μ::Float64
  ρ::Float64
  δ::Float64
  Kinematic::A
  function ARAP2D_regularized(; μ::Float64, ρ::Float64=0.0, δ::Float64=0.1, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(μ, ρ, δ, Kinematic)
  end


  function (obj::ARAP2D_regularized)(Λ::Float64=1.0)
    _, H, J_ = get_Kinematics(obj.Kinematic; Λ=Λ)
    μ, δ = obj.μ, obj.δ

    J(F) = 0.5 * (J_(F) + sqrt(J_(F)^2 + δ^2))
    ∂J(F) = 0.5 * (1.0 + J_(F) / sqrt(J_(F)^2 + δ^2))
    ∂2J(F) = 0.5 * δ^2 / ((J_(F)^2 + δ^2)^(3 / 2))


    J1 = 0.5 * (1.0 + sqrt(1.0 + δ^2))
    ∂J1 = 0.5 * (1.0 + 1.0 / sqrt(1.0^2 + δ^2))
    β = μ * (J1^(-1) - J1^(-2) * ∂J1)
    Ψ(F) = μ * 0.5 * J(F)^(-1) * (tr((F)' * F)) - β * log(J_(F))


    ∂Ψ1_∂J(F) = -μ / 2 * (tr((F)' * F)) * J(F)^(-2)
    ∂Ψ2_∂J(F) = -β / J_(F)
    ∂Ψ_∂J(F) = ∂Ψ1_∂J(F) * ∂J(F) + ∂Ψ2_∂J(F)
    ∂Ψ_∂F(F) = μ * F * J(F)^(-1)

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)
    I_ = I4()
    ∂Ψ1_∂J2(F) = μ * J(F)^(-3) * (tr((F)' * F))
    ∂Ψ2_∂J2(F) = β / J_(F)^2
    ∂Ψ_∂J2(F) = (∂Ψ1_∂J2(F) * ∂J(F)^2 + ∂Ψ1_∂J(F) * ∂2J(F)) + ∂Ψ2_∂J2(F)
    ∂Ψ_∂FJ(F) = -μ * J(F)^(-2) * F * ∂J(F)
    ∂Ψ_∂FF(F) = μ * J(F)^(-1) * I_

    ∂Ψuu(F) = ∂Ψ_∂FF(F) + ∂Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ_∂FJ(F) + ∂Ψ_∂J(F) * _∂H∂F_2D()
    return (Ψ, ∂Ψu, ∂Ψuu)
  end

end


struct ARAP2D{A} <: Mechano
  μ::Float64
  ρ::Float64
  Kinematic::A
  function ARAP2D(; μ::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(μ, ρ, Kinematic)
  end

  function (obj::ARAP2D)(Λ::Float64=1.0)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    μ = obj.μ
    I_ = I4()

    Ψ(F) = μ * 0.5 * J(F)^(-1) * (tr((F)' * F))
    ∂Ψ_∂F(F) = μ * F * J(F)^(-1)
    ∂Ψ_∂J(F) = -μ / 2 * (tr((F)' * F)) * J(F)^(-2)

    ∂2Ψ_∂J2(F) = μ * J(F)^(-3) * (tr((F)' * F))
    ∂2Ψ_∂FJ(F) = -μ * J(F)^(-2) * F
    ∂2Ψ_∂FF(F) = μ * J(F)^(-1) * I_


    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)
    ∂Ψuu(F) = ∂2Ψ_∂FF(F) + ∂2Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂2Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂2Ψ_∂FJ(F) + ∂Ψ_∂J(F) * _∂H∂F_2D()

    return (Ψ, ∂Ψu, ∂Ψuu)
  end

end


struct IncompressibleNeoHookean3D_2dP{A} <: Mechano
  μ::Float64
  ρ::Float64
  Kinematic::A

  function IncompressibleNeoHookean3D_2dP(; μ::Float64, τ::Float64, Δt::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(μ, τ, Δt, ρ, Kinematic)
  end

  function (obj::IncompressibleNeoHookean3D_2dP)(Λ::Float64=1.0; Threshold=0.01)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    μ = obj.μ
    I3_ = I3()
    Ψ(Ce) = μ / 2 * tr(Ce) * (det(Ce))^(-1 / 3)
    ∂Ψ∂Ce(Ce) =  μ / 2 * I3_ * (det(Ce))^(-1 / 3)
    ∂Ψ∂dCe(Ce) = - μ / 6 * tr(Ce) * (det(Ce))^(-4 / 3)
    Se(Ce)  = 2 * (∂Ψ∂Ce(Ce) + ∂Ψ∂dCe(Ce) * H(Ce)) 
    ∂2Ψ∂CedCe(Ce) =  - μ / 6 * I3_ * (det(Ce))^(-4 / 3)
    ∂2Ψ∂2dCe(Ce) =  2*μ / 9 * tr(Ce) * (det(Ce))^(-7 / 3)
    ∂Se∂Ce(Ce) = 2 *  (∂2Ψ∂2dCe(Ce) * (H(Ce) ⊗ H(Ce)) + ∂2Ψ∂CedCe(Ce) ⊗ H(Ce) + H(Ce) ⊗ ∂2Ψ∂CedCe(Ce) + ∂Ψ∂dCe(Ce) * ×ᵢ⁴(Ce))

    return (Ψ, Se, ∂Se∂Ce)

  end


end





# ===================
# MultiPhysicalModel models
# ===================


struct FlexoElectroModel{A} <: FlexoElectro
  ElectroMechano::A
  κ::Float64
  function FlexoElectroModel(; Mechano::Mechano, Electro::Electro, κ=1.0)
    physmodel = ElectroMechModel(Mechano=Mechano, Electro=Electro)
    A = typeof(physmodel)
    new{A}(physmodel, κ)
  end
  function (obj::FlexoElectroModel)(Λ::Float64=1.0)
    e₁ = VectorValue(1.0, 0.0, 0.0)
    e₂ = VectorValue(0.0, 1.0, 0.0)
    e₃ = VectorValue(0.0, 0.0, 1.0)
    # Φ(ϕ₁,ϕ₂,ϕ₃)=ϕ₁ ⊗₁² e₁+ϕ₂ ⊗₁² e₂+ϕ₃ ⊗₁² e₃
    f1(δϕ) = δϕ ⊗₁² e₁
    f2(δϕ) = δϕ ⊗₁² e₂
    f3(δϕ) = δϕ ⊗₁² e₃
    Φ(ϕ₁, ϕ₂, ϕ₃) = (f1 ∘ (ϕ₁) + f2 ∘ (ϕ₂) + f3 ∘ (ϕ₃))

    Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = obj.ElectroMechano(Λ)
    return Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ, Φ
  end

end

struct ElectroMechModel{A,B} <: ElectroMechano
  Mechano::A
  Electro::B
  function ElectroMechModel(; Mechano::Mechano, Electro::Electro)
    A, B = typeof(Mechano), typeof(Electro)
    new{A,B}(Mechano, Electro)
  end
  function (obj::ElectroMechModel)(Λ::Float64=1.0)
    Ψm, ∂Ψm_u, ∂Ψm_uu = obj.Mechano(Λ)
    Ψem, ∂Ψem_u, ∂Ψem_φ, ∂Ψem_uu, ∂Ψem_φu, ∂Ψem_φφ = _getCoupling(obj.Mechano, obj.Electro, Λ)
    Ψ(F, E) = Ψm(F) + Ψem(F, E)
    ∂Ψu(F, E) = ∂Ψm_u(F) + ∂Ψem_u(F, E)
    ∂Ψφ(F, E) = ∂Ψem_φ(F, E)
    ∂Ψuu(F, E) = ∂Ψm_uu(F) + ∂Ψem_uu(F, E)
    ∂Ψφu(F, E) = ∂Ψem_φu(F, E)
    ∂Ψφφ(F, E) = ∂Ψem_φφ(F, E)
    return (Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ)
  end

end

struct ThermoMechModel{A,B,C,D} <: ThermoMechano
  Thermo::A
  Mechano::B
  fθ::C
  dfdθ::D
  function ThermoMechModel(; Thermo::Thermo, Mechano::Mechano, fθ::Function, dfdθ::Function)
    A, B, C, D = typeof(Thermo), typeof(Mechano), typeof(fθ), typeof(dfdθ)
    new{A,B,C,D}(Thermo, Mechano, fθ, dfdθ)
  end

  function (obj::ThermoMechModel)(Λ::Float64=1.0)
    Ψt, ∂Ψt_θ, ∂Ψt_θθ = obj.Thermo(Λ)
    Ψm, ∂Ψm_u, ∂Ψm_uu = obj.Mechano(Λ)
    Ψtm, ∂Ψtm_u, ∂Ψtm_θ, ∂Ψtm_uu, ∂Ψtm_uθ, ∂Ψtm_θθ = _getCoupling(obj.Mechano, obj.Thermo, Λ)
    f(δθ) = (obj.fθ(δθ)::Float64)
    df(δθ) = (obj.dfdθ(δθ)::Float64)
    Ψ(F, δθ) = f(δθ) * (Ψm(F)) + (Ψt(δθ) + Ψtm(F, δθ))
    ∂Ψu(F, δθ) = f(δθ) * (∂Ψm_u(F)) + ∂Ψtm_u(F, δθ)
    ∂Ψθ(F, δθ) = df(δθ) * (Ψm(F)) + ∂Ψtm_θ(F, δθ) + ∂Ψt_θ(δθ)
    ∂Ψuu(F, δθ) = f(δθ) * (∂Ψm_uu(F)) + ∂Ψtm_uu(F, δθ)
    ∂Ψθθ(F, δθ) = ∂Ψtm_θθ(F, δθ) + ∂Ψt_θθ(δθ)
    ∂Ψuθ(F, δθ) = df(δθ) * (∂Ψm_u(F)) + ∂Ψtm_uθ(F, δθ)
    η(F, δθ) = -∂Ψθ(F, δθ)
    return (Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ, η)
  end


end

struct ThermoMech_EntropicPolyconvex{A,B,C,D,E} <: ThermoMechano
  Thermo::A
  Mechano::B
  β::Float64
  G::C
  ϕ::D
  s::E
  function ThermoMech_EntropicPolyconvex(; Thermo::Thermo, Mechano::Mechano, β::Float64, G::Function, ϕ::Function, s::Function)
    A, B, C, D, E = typeof(Thermo), typeof(Mechano), typeof(G), typeof(ϕ), typeof(s)
    new{A,B,C,D,E}(Thermo, Mechano, β, G, ϕ, s)
  end

  function (obj::ThermoMech_EntropicPolyconvex)(Λ::Float64=1.0)
    Ψt, _, _ = obj.Thermo(Λ)
    Ψm, _, _ = obj.Mechano(Λ)
    θr = obj.Thermo.θr
    Cv = obj.Thermo.Cv
    α = obj.Thermo.α
    β = obj.β
    G = obj.G
    ϕ = obj.ϕ
    s = obj.s

    _, H, J = get_Kinematics(obj.Mechano.Kinematic; Λ=Λ)

    I1(F) = tr(F' * F)
    I2(F) = tr(H(F)' * H(F))
    I3(F) = J(F)

    f(δθ) = (δθ + θr) / θr
    eᵣ(F) = α * (J(F) - 1.0)
    L1(δθ) = (1 - β) * Ψt(δθ)
    L2(δθ) = Cv * θr * (1 - β) * G(f(δθ))
    L3(F, δθ) = -Cv * θr * β * s(I1(F), I2(F), I3(F)) * ϕ(f(δθ))

    Ψ(F, δθ) = f(δθ) * Ψm(F) + (1 - f(δθ)) * eᵣ(F) + L1(δθ) + L2(δθ) + L3(F, δθ)

    ∂Ψ_∂∇u(F, δθ) = ForwardDiff.gradient(F -> Ψ(F, δθ), get_array(F))
    ∂Ψ_∂θ(F, δθ) = ForwardDiff.derivative(δθ -> Ψ(get_array(F), δθ), δθ)
    ∂2Ψ_∂2∇u(F, δθ) = ForwardDiff.hessian(F -> Ψ(F, δθ), get_array(F))
    ∂2Ψ_∂2θθ(F, δθ) = ForwardDiff.derivative(δθ -> ∂Ψ_∂θ(get_array(F), δθ), δθ)
    ∂2Ψ_∂2∇uθ(F, δθ) = ForwardDiff.derivative(δθ -> ∂Ψ_∂∇u(get_array(F), δθ), δθ)

    ∂Ψu(F, δθ) = TensorValue(∂Ψ_∂∇u(F, δθ))
    ∂Ψθ(F, δθ) = ∂Ψ_∂θ(F, δθ)
    ∂Ψuu(F, δθ) = TensorValue(∂2Ψ_∂2∇u(F, δθ))
    ∂Ψθθ(F, δθ) = ∂2Ψ_∂2θθ(F, δθ)
    ∂Ψuθ(F, δθ) = TensorValue(∂2Ψ_∂2∇uθ(F, δθ))

    return (Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ)
  end
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


struct MagnetoVacuumModel{A} <: MagnetoMechano
  Magneto::A
  function MagnetoVacuumModel(; Magneto::Magneto)
    A = typeof(Magneto)
    new{A}(Magneto)
  end

  function (obj::MagnetoVacuumModel)(Λ::Float64=1.0)

    _, H, J = get_Kinematics(Kinematics(Mechano); Λ=Λ)

    μ, χe = obj.Magneto.μ, obj.Magneto.χe

    # Energy #
    Hℋ₀(F, ℋ₀) = H(F) * ℋ₀
    Hℋ₀Hℋ₀(F, ℋ₀) = Hℋ₀(F, ℋ₀) ⋅ Hℋ₀(F, ℋ₀)
    Ψmm(F, ℋ₀) = (-μ / (2.0 * J(F))) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)

    # First Derivatives #
    I2_ = I2()
    ∂Ψmm_∂H(F, ℋ₀) = (-μ / (J(F))) * (Hℋ₀(F, ℋ₀) ⊗ ℋ₀) * (1 + χe)
    ∂Ψmm_∂J(F, ℋ₀) = (+μ / (2.0 * J(F)^2.0)) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)
    ∂Ψmm_∂ℋ₀(F, ℋ₀) = (-μ / (J(F))) * (H(F)' * Hℋ₀(F, ℋ₀)) * (1 + χe)
    ∂Ψmm_∂u(F, ℋ₀) = (tr(∂Ψmm_∂H(F, ℋ₀)) * I2_) - ∂Ψmm_∂H(F, ℋ₀)' + ∂Ψmm_∂J(F, ℋ₀) * H(F)
    ∂Ψmm_∂φ(F, ℋ₀) = ∂Ψmm_∂ℋ₀(F, ℋ₀)

    # Second Derivatives #
    ∂Ψmm_∂HH(F, ℋ₀) = (-μ / (J(F))) * (I2_ ⊗₁₃²⁴ (ℋ₀ ⊗ ℋ₀)) * (1 + χe)
    ∂Ψmm_∂HJ(F, ℋ₀) = (+μ / (J(F))^2.0) * (Hℋ₀(F, ℋ₀) ⊗ ℋ₀) * (1 + χe)
    ∂Ψmm_∂JJ(F, ℋ₀) = (-μ / (J(F))^3.0) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)
    ∂Ψmm_∂uu(F, ℋ₀) = _∂H∂F_2D()' * ∂Ψmm_∂HH(F, ℋ₀) * _∂H∂F_2D() + _∂H∂F_2D()' * (∂Ψmm_∂HJ(F, ℋ₀) ⊗ H(F)) +
                      (H(F) ⊗ ∂Ψmm_∂HJ(F, ℋ₀)) * _∂H∂F_2D() + ∂Ψmm_∂JJ(F, ℋ₀) * (H(F) ⊗ H(F)) + ∂Ψmm_∂J(F, ℋ₀) * _∂H∂F_2D()


    ∂Ψmm_∂ℋ₀H(F, ℋ₀) = (-μ / (J(F))) * ((I2_ ⊗₁₃² Hℋ₀(F, ℋ₀)) + (H(F)' ⊗₁₂³ ℋ₀)) * (1 + χe)
    ∂Ψmm_∂ℋ₀J(F, ℋ₀) = (+μ / (J(F))^2.0) * (H(F)' * Hℋ₀(F, ℋ₀)) * (1 + χe)
    ∂Ψmm_∂φu(F, ℋ₀) = ∂Ψmm_∂ℋ₀H(F, ℋ₀) * _∂H∂F_2D() + (∂Ψmm_∂ℋ₀J(F, ℋ₀) ⊗₁²³ H(F))
    ∂Ψmm_∂φφ(F, ℋ₀) = (-μ / (J(F))) * (H(F)' * H(F)) * (1 + χe)


    return (Ψmm, ∂Ψmm_∂u, ∂Ψmm_∂φ, ∂Ψmm_∂uu, ∂Ψmm_∂φu, ∂Ψmm_∂φφ)
  end



end


# ===============================
# Coupling terms for multiphysic
# ===============================

function _getCoupling(mec::Mechano, elec::IdealDielectric, Λ::Float64)
  _, H, J = get_Kinematics(mec.Kinematic; Λ=Λ)

  # Energy #
  HE(F, E) = H(F) * E
  HEHE(F, E) = HE(F, E) ⋅ HE(F, E)
  Ψem(F, E) = (-elec.ε / (2.0 * J(F))) * HEHE(F, E)
  # First Derivatives #
  ∂Ψem_∂H(F, E) = (-elec.ε / (J(F))) * (HE(F, E) ⊗ E)
  ∂Ψem_∂J(F, E) = (+elec.ε / (2.0 * J(F)^2.0)) * HEHE(F, E)
  ∂Ψem_∂E(F, E) = (-elec.ε / (J(F))) * (H(F)' * HE(F, E))
  ∂Ψem_u(F, E) = ∂Ψem_∂H(F, E) × F + ∂Ψem_∂J(F, E) * H(F)
  # ∂Ψem_φ(F, E) = -∂Ψem_∂E(F, E)
  ∂Ψem_φ(F, E) = ∂Ψem_∂E(F, E)

  # Second Derivatives #
  # I33 = TensorValue(Matrix(1.0I, 3, 3))
  I33 = I3()
  ∂Ψem_HH(F, E) = (-elec.ε / (J(F))) * (I33 ⊗₁₃²⁴ (E ⊗ E))
  ∂Ψem_HJ(F, E) = (+elec.ε / (J(F))^2.0) * (HE(F, E) ⊗ E)
  ∂Ψem_JJ(F, E) = (-elec.ε / (J(F))^3.0) * HEHE(F, E)
  ∂Ψem_uu(F, E) = (F × (∂Ψem_HH(F, E) × F)) +
                  H(F) ⊗₁₂³⁴ (∂Ψem_HJ(F, E) × F) +
                  (∂Ψem_HJ(F, E) × F) ⊗₁₂³⁴ H(F) +
                  ∂Ψem_JJ(F, E) * (H(F) ⊗₁₂³⁴ H(F)) +
                  ×ᵢ⁴(∂Ψem_∂H(F, E) + ∂Ψem_∂J(F, E) * F)

  ∂Ψem_EH(F, E) = (-elec.ε / (J(F))) * ((I33 ⊗₁₃² HE(F, E)) + (H(F)' ⊗₁₂³ E))
  ∂Ψem_EJ(F, E) = (+elec.ε / (J(F))^2.0) * (H(F)' * HE(F, E))

  # ∂Ψem_φu(F, E) = -(∂Ψem_EH(F, E) × F) - (∂Ψem_EJ(F, E) ⊗₁²³ H(F))
  ∂Ψem_φu(F, E) = (∂Ψem_EH(F, E) × F) + (∂Ψem_EJ(F, E) ⊗₁²³ H(F))

  ∂Ψem_φφ(F, E) = (-elec.ε / (J(F))) * (H(F)' * H(F))

  return (Ψem, ∂Ψem_u, ∂Ψem_φ, ∂Ψem_uu, ∂Ψem_φu, ∂Ψem_φφ)

end

function _getCoupling(mec::Mechano, term::Thermo, Λ::Float64)
  _, H, J = get_Kinematics(mec.Kinematic; Λ=Λ)

  ∂Ψtm_∂J(F, δθ) = -6.0 * term.α * J(F) * δθ
  ∂Ψtm_u(F, δθ) = ∂Ψtm_∂J(F, δθ) * H(F)
  ∂Ψtm_θ(F, δθ) = -3.0 * term.α * (J(F)^2.0 - 1.0)
  ∂Ψtm_uu(F, δθ) = (-6.0 * term.α * δθ) * (H(F) ⊗₁₂³⁴ H(F)) + ×ᵢ⁴(∂Ψtm_∂J(F, δθ) * F)
  ∂Ψtm_uθ(F, δθ) = -6.0 * term.α * J(F) * H(F)
  ∂Ψtm_θθ(F, δθ) = 0.0

  Ψtm(F, δθ) = ∂Ψtm_θ(F, δθ) * δθ

  return (Ψtm, ∂Ψtm_u, ∂Ψtm_θ, ∂Ψtm_uu, ∂Ψtm_uθ, ∂Ψtm_θθ)
end

function _getCoupling(mec::Mechano, mag::IdealMagnetic, Λ::Float64)
  _, H, J = get_Kinematics(mec.Kinematic; Λ=Λ)
  μ, χe = mag.μ, mag.χe

  # Energy #
  Hℋ₀(F, ℋ₀) = H(F) * ℋ₀
  Hℋ₀Hℋ₀(F, ℋ₀) = Hℋ₀(F, ℋ₀) ⋅ Hℋ₀(F, ℋ₀)
  Ψmm(F, ℋ₀) = (-μ / (2.0 * J(F))) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)

  # First Derivatives #
  ∂Ψmm_∂H(F, ℋ₀) = (-μ / (J(F))) * (Hℋ₀(F, ℋ₀) ⊗ ℋ₀) * (1 + χe)
  ∂Ψmm_∂J(F, ℋ₀) = (+μ / (2.0 * J(F)^2.0)) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)
  ∂Ψmm_∂ℋ₀(F, ℋ₀) = (-μ / (J(F))) * (H(F)' * Hℋ₀(F, ℋ₀)) * (1 + χe)
  ∂Ψmm_∂u(F, ℋ₀) = ∂Ψmm_∂H(F, ℋ₀) × F + ∂Ψmm_∂J(F, ℋ₀) * H(F)
  ∂Ψmm_∂φ(F, ℋ₀) = ∂Ψmm_∂ℋ₀(F, ℋ₀)

  # Second Derivatives #
  I33 = I3()
  ∂Ψmm_∂HH(F, ℋ₀) = (-μ / (J(F))) * (I33 ⊗₁₃²⁴ (ℋ₀ ⊗ ℋ₀)) * (1 + χe)
  ∂Ψmm_∂HJ(F, ℋ₀) = (+μ / (J(F))^2.0) * (Hℋ₀(F, ℋ₀) ⊗ ℋ₀) * (1 + χe)
  ∂Ψmm_∂JJ(F, ℋ₀) = (-μ / (J(F))^3.0) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)
  ∂Ψmm_∂uu(F, ℋ₀) = (F × (∂Ψmm_∂HH(F, ℋ₀) × F)) +
                    H(F) ⊗₁₂³⁴ (∂Ψmm_∂HJ(F, ℋ₀) × F) +
                    (∂Ψmm_∂HJ(F, ℋ₀) × F) ⊗₁₂³⁴ H(F) +
                    ∂Ψmm_∂JJ(F, ℋ₀) * (H(F) ⊗₁₂³⁴ H(F)) +
                    ×ᵢ⁴(∂Ψmm_∂H(F, ℋ₀) + ∂Ψmm_∂J(F, ℋ₀) * F)

  ∂Ψmm_∂ℋ₀H(F, ℋ₀) = (-μ / (J(F))) * ((I33 ⊗₁₃² Hℋ₀(F, ℋ₀)) + (H(F)' ⊗₁₂³ ℋ₀)) * (1 + χe)
  ∂Ψmm_∂ℋ₀J(F, ℋ₀) = (+μ / (J(F))^2.0) * (H(F)' * Hℋ₀(F, ℋ₀)) * (1 + χe)

  ∂Ψmm_∂φu(F, ℋ₀) = (∂Ψmm_∂ℋ₀H(F, ℋ₀) × F) + (∂Ψmm_∂ℋ₀J(F, ℋ₀) ⊗₁²³ H(F))
  ∂Ψmm_∂φφ(F, ℋ₀) = (-μ / (J(F))) * (H(F)' * H(F)) * (1 + χe)


  Ψ(F, ℋ₀, N) = Ψmm(F, ℋ₀)
  ∂Ψ_u(F, ℋ₀, N) = ∂Ψmm_∂u(F, ℋ₀)
  ∂Ψ_φ(F, ℋ₀, N) = ∂Ψmm_∂φ(F, ℋ₀)
  ∂Ψ_uu(F, ℋ₀, N) = ∂Ψmm_∂uu(F, ℋ₀)
  ∂Ψ_φu(F, ℋ₀, N) = ∂Ψmm_∂φu(F, ℋ₀)
  ∂Ψ_φφ(F, ℋ₀, N) = ∂Ψmm_∂φφ(F, ℋ₀)

  return (Ψ, ∂Ψ_u, ∂Ψ_φ, ∂Ψ_uu, ∂Ψ_φu, ∂Ψ_φφ)

end

function _getCoupling(mec::Mechano, mag::IdealMagnetic2D, Λ::Float64)
  _, H, J = get_Kinematics(mec.Kinematic; Λ=Λ)

  μ, χe = mag.μ, mag.χe

  # Energy #
  Hℋ₀(F, ℋ₀) = H(F) * ℋ₀
  Hℋ₀Hℋ₀(F, ℋ₀) = Hℋ₀(F, ℋ₀) ⋅ Hℋ₀(F, ℋ₀)
  Ψmm(F, ℋ₀) = (-μ / (2.0 * J(F))) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)

  # First Derivatives #
  I2_ = I2()
  ∂Ψmm_∂H(F, ℋ₀) = (-μ / (J(F))) * (Hℋ₀(F, ℋ₀) ⊗ ℋ₀) * (1 + χe)
  ∂Ψmm_∂J(F, ℋ₀) = (+μ / (2.0 * J(F)^2.0)) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)
  ∂Ψmm_∂ℋ₀(F, ℋ₀) = (-μ / (J(F))) * (H(F)' * Hℋ₀(F, ℋ₀)) * (1 + χe)
  ∂Ψmm_∂u(F, ℋ₀) = (tr(∂Ψmm_∂H(F, ℋ₀)) * I2_) - ∂Ψmm_∂H(F, ℋ₀)' + ∂Ψmm_∂J(F, ℋ₀) * H(F)
  ∂Ψmm_∂φ(F, ℋ₀) = ∂Ψmm_∂ℋ₀(F, ℋ₀)

  # Second Derivatives #
  ∂Ψmm_∂HH(F, ℋ₀) = (-μ / (J(F))) * (I2_ ⊗₁₃²⁴ (ℋ₀ ⊗ ℋ₀)) * (1 + χe)
  ∂Ψmm_∂HJ(F, ℋ₀) = (+μ / (J(F))^2.0) * (Hℋ₀(F, ℋ₀) ⊗ ℋ₀) * (1 + χe)
  ∂Ψmm_∂JJ(F, ℋ₀) = (-μ / (J(F))^3.0) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)
  ∂Ψmm_∂uu(F, ℋ₀) = _∂H∂F_2D()' * ∂Ψmm_∂HH(F, ℋ₀) * _∂H∂F_2D() + _∂H∂F_2D()' * (∂Ψmm_∂HJ(F, ℋ₀) ⊗ H(F)) +
                    (H(F) ⊗ ∂Ψmm_∂HJ(F, ℋ₀)) * _∂H∂F_2D() + ∂Ψmm_∂JJ(F, ℋ₀) * (H(F) ⊗ H(F)) + ∂Ψmm_∂J(F, ℋ₀) * _∂H∂F_2D()


  ∂Ψmm_∂ℋ₀H(F, ℋ₀) = (-μ / (J(F))) * ((I2_ ⊗₁₃² Hℋ₀(F, ℋ₀)) + (H(F)' ⊗₁₂³ ℋ₀)) * (1 + χe)
  ∂Ψmm_∂ℋ₀J(F, ℋ₀) = (+μ / (J(F))^2.0) * (H(F)' * Hℋ₀(F, ℋ₀)) * (1 + χe)
  ∂Ψmm_∂φu(F, ℋ₀) = ∂Ψmm_∂ℋ₀H(F, ℋ₀) * _∂H∂F_2D() + (∂Ψmm_∂ℋ₀J(F, ℋ₀) ⊗₁²³ H(F))
  ∂Ψmm_∂φφ(F, ℋ₀) = (-μ / (J(F))) * (H(F)' * H(F)) * (1 + χe)


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
  # Energy #
  Hℋ₀(F, ℋ₀) = H(F) * ℋ₀
  Hℋ₀Hℋ₀(F, ℋ₀) = Hℋ₀(F, ℋ₀) ⋅ Hℋ₀(F, ℋ₀)
  Ψmm(F, ℋ₀) = (-μ / (2.0 * J(F))) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)

  # First Derivatives #
  ∂Ψmm_∂H(F, ℋ₀) = (-μ / (J(F))) * (Hℋ₀(F, ℋ₀) ⊗ ℋ₀) * (1 + χe)
  ∂Ψmm_∂J(F, ℋ₀) = (+μ / (2.0 * J(F)^2.0)) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)
  ∂Ψmm_∂ℋ₀(F, ℋ₀) = (-μ / (J(F))) * (H(F)' * Hℋ₀(F, ℋ₀)) * (1 + χe)
  ∂Ψmm_∂u(F, ℋ₀) = ∂Ψmm_∂H(F, ℋ₀) × F + ∂Ψmm_∂J(F, ℋ₀) * H(F)
  ∂Ψmm_∂φ(F, ℋ₀) = ∂Ψmm_∂ℋ₀(F, ℋ₀)

  # Second Derivatives #
  # I33 = TensorValue(Matrix(1.0I, 3, 3))
  I33 = I3()
  ∂Ψmm_∂HH(F, ℋ₀) = (-μ / (J(F))) * (I33 ⊗₁₃²⁴ (ℋ₀ ⊗ ℋ₀)) * (1 + χe)
  ∂Ψmm_∂HJ(F, ℋ₀) = (+μ / (J(F))^2.0) * (Hℋ₀(F, ℋ₀) ⊗ ℋ₀) * (1 + χe)
  ∂Ψmm_∂JJ(F, ℋ₀) = (-μ / (J(F))^3.0) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)
  ∂Ψmm_∂uu(F, ℋ₀) = (F × (∂Ψmm_∂HH(F, ℋ₀) × F)) +
                    H(F) ⊗₁₂³⁴ (∂Ψmm_∂HJ(F, ℋ₀) × F) +
                    (∂Ψmm_∂HJ(F, ℋ₀) × F) ⊗₁₂³⁴ H(F) +
                    ∂Ψmm_∂JJ(F, ℋ₀) * (H(F) ⊗₁₂³⁴ H(F)) +
                    ×ᵢ⁴(∂Ψmm_∂H(F, ℋ₀) + ∂Ψmm_∂J(F, ℋ₀) * F)

  ∂Ψmm_∂ℋ₀H(F, ℋ₀) = (-μ / (J(F))) * ((I33 ⊗₁₃² Hℋ₀(F, ℋ₀)) + (H(F)' ⊗₁₂³ ℋ₀)) * (1 + χe)
  ∂Ψmm_∂ℋ₀J(F, ℋ₀) = (+μ / (J(F))^2.0) * (H(F)' * Hℋ₀(F, ℋ₀)) * (1 + χe)

  ∂Ψmm_∂φu(F, ℋ₀) = (∂Ψmm_∂ℋ₀H(F, ℋ₀) × F) + (∂Ψmm_∂ℋ₀J(F, ℋ₀) ⊗₁²³ H(F))
  ∂Ψmm_∂φφ(F, ℋ₀) = (-μ / (J(F))) * (H(F)' * H(F)) * (1 + χe)


  #-------------------------------------------------------------------------------------
  # SECOND TERM
  #-------------------------------------------------------------------------------------

  ℋᵣ(N) = αr * N
  Fℋᵣ(F, N) = F * ℋᵣ(N)
  Ψcoup(F, N) = (μ * J(F)) * (Fℋᵣ(F, N) ⋅ Fℋᵣ(F, N) - ℋᵣ(N) ⋅ ℋᵣ(N))
  ∂Ψcoup_∂F(F, N) = 2 * (μ * J(F)) * (Fℋᵣ(F, N) ⊗ ℋᵣ(N))
  ∂Ψcoup_∂J(F, N) = (μ) * (Fℋᵣ(F, N) ⋅ Fℋᵣ(F, N) - ℋᵣ(N) ⋅ ℋᵣ(N))
  ∂Ψcoup_∂u(F, N) = ∂Ψcoup_∂J(F, N) * H(F) + ∂Ψcoup_∂F(F, N)

  ∂Ψcoup_∂JF(F, N) = 2 * μ * (H(F) ⊗ (Fℋᵣ(F, N) ⊗ ℋᵣ(N)) + (Fℋᵣ(F, N) ⊗ ℋᵣ(N)) ⊗ H(F))
  ∂Ψcoup_∂FF(F, N) = 2 * μ * J(F) * (I33 ⊗₁₃²⁴ (ℋᵣ(N) ⊗ ℋᵣ(N)))
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

  ∂Ψtorq_∂HH(F, ℋ₀, N) = (μ * (1 + χe) / J(F)) * (I33 ⊗₁₃²⁴ (ℋᵣ(N) ⊗ ℋ₀ + ℋ₀ ⊗ ℋᵣ(N)))
  ∂Ψtorq_∂HJ(F, ℋ₀, N) = -(μ * (1 + χe) / J(F)^2) * (Hℋᵣ(F, N) ⊗ ℋ₀ + Hℋ₀(F, ℋ₀) ⊗ ℋᵣ(N))
  ∂Ψtorq_∂JJ(F, ℋ₀, N) = (μ * (1 + χe) / J(F)^3) * (Hℋ₀(F, ℋ₀) ⋅ Hℋᵣ(F, N))

  ∂Ψtorq_∂uu(F, ℋ₀, N) = (F × (∂Ψtorq_∂HH(F, ℋ₀, N) × F)) +
                         H(F) ⊗₁₂³⁴ (∂Ψtorq_∂HJ(F, ℋ₀, N) × F) +
                         (∂Ψtorq_∂HJ(F, ℋ₀, N) × F) ⊗₁₂³⁴ H(F) +
                         ∂Ψtorq_∂JJ(F, ℋ₀, N) * (H(F) ⊗₁₂³⁴ H(F)) +
                         ×ᵢ⁴(∂Ψtorq_∂H(F, ℋ₀, N) + ∂Ψtorq_∂J(F, ℋ₀, N) * F)


  ∂Ψtorq_∂ℋ₀H(F, ℋ₀, N) = (μ / (J(F))) * ((I33 ⊗₁₃² Hℋᵣ(F, N)) + (H(F)' ⊗₁₂³ Hℋᵣ(F, N))) * (1 + χe)
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

  #-------------------------------------------------------------------------------------
  # FIRST TERM
  #-------------------------------------------------------------------------------------

  # Energy #
  Hℋ₀(F, ℋ₀) = H(F) * ℋ₀
  Hℋ₀Hℋ₀(F, ℋ₀) = Hℋ₀(F, ℋ₀) ⋅ Hℋ₀(F, ℋ₀)
  Ψmm(F, ℋ₀) = (-μ / (2.0 * J(F))) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)

  # First Derivatives #
  I2_ = I2()
  ∂Ψmm_∂H(F, ℋ₀) = (-μ / (J(F))) * (Hℋ₀(F, ℋ₀) ⊗ ℋ₀) * (1 + χe)
  ∂Ψmm_∂J(F, ℋ₀) = (+μ / (2.0 * J(F)^2.0)) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)
  ∂Ψmm_∂ℋ₀(F, ℋ₀) = (-μ / (J(F))) * (H(F)' * Hℋ₀(F, ℋ₀)) * (1 + χe)
  ∂Ψmm_∂u(F, ℋ₀) = (tr(∂Ψmm_∂H(F, ℋ₀)) * I2_) - ∂Ψmm_∂H(F, ℋ₀)' + ∂Ψmm_∂J(F, ℋ₀) * H(F)
  ∂Ψmm_∂φ(F, ℋ₀) = ∂Ψmm_∂ℋ₀(F, ℋ₀)

  # Second Derivatives #
  ∂Ψmm_∂HH(F, ℋ₀) = (-μ / (J(F))) * (I2_ ⊗₁₃²⁴ (ℋ₀ ⊗ ℋ₀)) * (1 + χe)
  ∂Ψmm_∂HJ(F, ℋ₀) = (+μ / (J(F))^2.0) * (Hℋ₀(F, ℋ₀) ⊗ ℋ₀) * (1 + χe)
  ∂Ψmm_∂JJ(F, ℋ₀) = (-μ / (J(F))^3.0) * Hℋ₀Hℋ₀(F, ℋ₀) * (1 + χe)
  ∂Ψmm_∂uu(F, ℋ₀) = _∂H∂F_2D()' * ∂Ψmm_∂HH(F, ℋ₀) * _∂H∂F_2D() + _∂H∂F_2D()' * (∂Ψmm_∂HJ(F, ℋ₀) ⊗ H(F)) +
                    (H(F) ⊗ ∂Ψmm_∂HJ(F, ℋ₀)) * _∂H∂F_2D() + ∂Ψmm_∂JJ(F, ℋ₀) * (H(F) ⊗ H(F)) + ∂Ψmm_∂J(F, ℋ₀) * _∂H∂F_2D()


  ∂Ψmm_∂ℋ₀H(F, ℋ₀) = (-μ / (J(F))) * ((I2_ ⊗₁₃² Hℋ₀(F, ℋ₀)) + (H(F)' ⊗₁₂³ ℋ₀)) * (1 + χe)
  ∂Ψmm_∂ℋ₀J(F, ℋ₀) = (+μ / (J(F))^2.0) * (H(F)' * Hℋ₀(F, ℋ₀)) * (1 + χe)
  ∂Ψmm_∂φu(F, ℋ₀) = ∂Ψmm_∂ℋ₀H(F, ℋ₀) * _∂H∂F_2D() + (∂Ψmm_∂ℋ₀J(F, ℋ₀) ⊗₁²³ H(F))
  ∂Ψmm_∂φφ(F, ℋ₀) = (-μ / (J(F))) * (H(F)' * H(F)) * (1 + χe)


  #-------------------------------------------------------------------------------------
  # SECOND TERM
  #-------------------------------------------------------------------------------------

  ℋᵣ(N) = αr * N
  Fℋᵣ(F, N) = F * ℋᵣ(N)
  Ψcoup(F, N) = (μ * J(F)) * (Fℋᵣ(F, N) ⋅ Fℋᵣ(F, N) - ℋᵣ(N) ⋅ ℋᵣ(N))
  ∂Ψcoup_∂F(F, N) = 2 * (μ * J(F)) * (Fℋᵣ(F, N) ⊗ ℋᵣ(N))
  ∂Ψcoup_∂J(F, N) = (μ) * (Fℋᵣ(F, N) ⋅ Fℋᵣ(F, N) - ℋᵣ(N) ⋅ ℋᵣ(N))
  ∂Ψcoup_∂u(F, N) = ∂Ψcoup_∂J(F, N) * H(F) + ∂Ψcoup_∂F(F, N)

  ∂Ψcoup_∂JF(F, N) = 2 * μ * (H(F) ⊗ (Fℋᵣ(F, N) ⊗ ℋᵣ(N)) + (Fℋᵣ(F, N) ⊗ ℋᵣ(N)) ⊗ H(F))
  ∂Ψcoup_∂FF(F, N) = 2 * μ * J(F) * (I2_ ⊗₁₃²⁴ (ℋᵣ(N) ⊗ ℋᵣ(N)))
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
  ∂Ψtorq_∂u(F, ℋ₀, N) = (tr(∂Ψtorq_∂H(F, ℋ₀, N)) * I2_) - ∂Ψtorq_∂H(F, ℋ₀, N)' + ∂Ψtorq_∂J(F, ℋ₀, N) * H(F)
  ∂Ψtorq_∂φ(F, ℋ₀, N) = (μ * (1 + χe) / J(F)) * (H(F)' * Hℋᵣ(F, N))

  ∂Ψtorq_∂HH(F, ℋ₀, N) = (μ * (1 + χe) / J(F)) * (I2_ ⊗₁₃²⁴ (ℋᵣ(N) ⊗ ℋ₀ + ℋ₀ ⊗ ℋᵣ(N)))
  ∂Ψtorq_∂HJ(F, ℋ₀, N) = -(μ * (1 + χe) / J(F)^2) * (Hℋᵣ(F, N) ⊗ ℋ₀ + Hℋ₀(F, ℋ₀) ⊗ ℋᵣ(N))
  ∂Ψtorq_∂JJ(F, ℋ₀, N) = (μ * (1 + χe) / J(F)^3) * (Hℋ₀(F, ℋ₀) ⋅ Hℋᵣ(F, N))

  ∂Ψtorq_∂uu(F, ℋ₀, N) = _∂H∂F_2D()' * ∂Ψtorq_∂HH(F, ℋ₀, N) * _∂H∂F_2D() +
                         _∂H∂F_2D()' * (∂Ψtorq_∂HJ(F, ℋ₀, N) ⊗ H(F)) +
                         (H(F) ⊗ ∂Ψtorq_∂HJ(F, ℋ₀, N)) * _∂H∂F_2D() +
                         ∂Ψtorq_∂JJ(F, ℋ₀, N) * (H(F) ⊗ H(F)) +
                         ∂Ψtorq_∂J(F, ℋ₀, N) * _∂H∂F_2D()

  ∂Ψtorq_∂ℋ₀H(F, ℋ₀, N) = (μ / (J(F))) * ((I2_ ⊗₁₃² Hℋᵣ(F, N)) + (H(F)' ⊗₁₂³ Hℋᵣ(F, N))) * (1 + χe)
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