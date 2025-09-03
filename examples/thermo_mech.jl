
#-----------------------------------------------
#-----------------------------------------------
# This script is going to be used in order to generate
# the following type of target configuration:
#
# function utarget_(Lx, α, L, n)
# A = α * Lx
# θ = -10*(π/180)
# R = [[cos(θ) 0 -sin(θ)];[0 1 0];[sin(θ) 0 cos(θ)]]
# xtarget(x) = [A * sin(n * π * x[1] / Lx), x[2], x[1] / Lx * L]
# return (x) -> VectorValue(R*xtarget(x) - [x[1],x[2],x[3]])
#end
#
# I will be varying α, L and n in this study
#
#-----------------------------------------------
#-----------------------------------------------

using Gridap, GridapGmsh, GridapSolvers, DrWatson, TimerOutputs
using GridapSolvers.NonlinearSolvers
using GridapSolvers.LinearSolvers
using TimerOutputs

using Gridap.FESpaces
using HyperFEM: jacobian, IterativeSolver, solve!
using WriteVTK
using Revise


using Gridap.MultiField
# using GridapDistributed
using Gridap.FESpaces: get_assembly_strategy
using Gridap.Algebra
# using BlockArrays
using HyperFEM
# using HyperShape
# using HyperShape: evaluate!
using ForwardDiff
using LinearAlgebra

using Base.Threads



μParams = [0.010000000052968838, 12480.64495232286,
1.999999975065904, 1.9999999999998976,
5195.545287237134, 0.2602717127043121, 1.9999999999999953]


physmodel = MooneyRivlin3D(λ=0.0, μ1=μParams[1], μ2=μParams[2])
TransIso = TransverseIsotropy3D(μ=μParams[5], α=μParams[6], β=μParams[7])
Ψ1, ∂Ψu1, ∂Ψuu1 = physmodel()
Ψ2, ∂Ψu2, ∂Ψuu2 = TransIso()
ΨR = (F, N) -> Ψ1(F) + Ψ2(F, N)
∂ΨRu = (F, N) -> ∂Ψu1(F) + ∂Ψu2(F, N)
∂ΨRuu = (F, N) -> ∂Ψuu1(F) + ∂Ψuu2(F, N)

modvol = MooneyRivlin3D(λ=(μParams[1]+μParams[2])*100, μ1=0.0, μ2=0.0)


F = TensorValue(rand(3,3))
N = VectorValue(rand(3))

∂ΨRu(F,N)



struct ThermoJavierGil <: Thermo
  θr::Float64
  γ::Float64
  function (obj::ThermoJavierGil)(θR::Float64)
    g(θ) = 1/(obj.γ+1) * ((θ/obj.θR)^(obj.γ+1) -1)
    dg(θ) = θ^obj.γ / obj.θR^(obj.γ+1)
    ddg(θ) = θ^(obj.γ-1) / obj.θR^(obj.γ+1)
    return g, dg, ddg
  end
end



struct ThermalExpansion{A} <: Thermo
  λ::Float64
  μ1::Float64
  μ2::Float64
  ρ::Float64
  Kinematic::A
  function MooneyRivlin3D(; λ::Float64, μ1::Float64, μ2::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ1, μ2, ρ, Kinematic)
  end
  function (obj::MooneyRivlin3D)(Λ::Float64=1.0; Threshold=0.01)
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


struct ThermoMechJavierGil
    Deviatoric::Mechano
    Isochoric::Mechano # ElectroMechModel
    Thermo::Thermo

end