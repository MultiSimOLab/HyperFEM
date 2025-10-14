
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

    ∂2Ψ(F, Jh) = begin
      vecval = eigen(get_array(∂2Ψ_(F, Jh)))
      vec = real(vecval.vectors)
      val = real(vecval.values)
      TensorValue(vec * diagm(max.(δ, val)) * vec')
    end
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


struct ComposedElasticModel <: Elasto
  Model1::Elasto
  Model2::Elasto
  Kinematic
  function ComposedElasticModel(model1::Elasto,model2::Elasto)
    @assert model1.Kinematic == model2.Kinematic
    new(model1,model2,model1.Kinematic)
  end
  function (obj::ComposedElasticModel)(Λ::Float64=1.0)
    DΨ1 = obj.Model1(Λ)
    DΨ2 = obj.Model2(Λ)
    Ψ, ∂Ψ, ∂∂Ψ = map((ψ1,ψ2) -> (x) -> ψ1(x) + ψ2(x), DΨ1, DΨ2)
    return (Ψ, ∂Ψ, ∂∂Ψ)
  end
end

function (+)(Model1::Elasto, Model2::Elasto)
  ComposedElasticModel(Model1,Model2)
end


# ===================
# Mechanical models
# ===================

struct Yeoh3D{A} <: Mechano
    λ::Float64
    C10::Float64
    C20::Float64
    C30::Float64
    Kinematic::A
    function Yeoh3D(; λ::Float64, C10::Float64, C20::Float64, C30::Float64, Kinematic::KinematicModel=Kinematics(Mechano))
        new{typeof(Kinematic)}(λ, C10, C20, C30, Kinematic)
    end

    function (obj::Yeoh3D)(Λ::Float64=1.0; Threshold=0.01)
        _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
        λ, C10, C20, C30 = obj.λ, obj.C10, obj.C20, obj.C30

        # Free energy
        I1(F) = tr((F)' * F)
        ψ(F)  = C10 * (I1(F) - 3) + C20 * (I1(F) - 3)^2 + C30 * (I1(F) - 3)^3 -2*C10*log(J(F)) + 0.5*λ*(J(F)-1)^2

        # First Piola-Kirchhoff
        ∂ψ_∂I1(F) = C10 + 2*C20*(I1(F) - 3) + 3*C30*(I1(F) - 3)^2
        ∂log∂J(J) = J >= Threshold ? 1 / J : (2 / Threshold - J / (Threshold^2))
        ∂ψ_∂J(F)  = -2*C10 * ∂log∂J(J(F)) + λ*(J(F) - 1)
        ∂ψu(F)    = 2*∂ψ_∂I1(F)*F + ∂ψ_∂J(F)*H(F)
        
        # Elasticity
        ∂2ψ_∂I1I1(F) = 2*C20 + 6*C30*(I1(F)-3)
        ∂log2∂J2(J)  = J >= Threshold ? -1 / (J^2) : (-1 / (Threshold^2))
        ∂2ψ_∂JJ(F)   = -2*C10*∂log2∂J2(J(F)) + λ
        ∂ψuu(F)      = 4*∂2ψ_∂I1I1(F)*(F ⊗ F) + 2*∂ψ_∂I1(F)*I9 + ∂2ψ_∂JJ(F)*(H(F) ⊗ H(F)) + ∂ψ_∂J(F)*(I9 × F)


        return (ψ, ∂ψu, ∂ψuu)
    end
end

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
    ε(F) = 0.5(F + F') -I2
    ∂Ψuu(F) = μ * (δᵢₖδⱼₗ2D + δᵢₗδⱼₖ2D) + λ * δᵢⱼδₖₗ2D
    ∂Ψu(F) = ∂Ψuu(F) ⊙ (F - I2)
    Ψ(F) = μ * sum(ε(F).*ε(F)) + 0.5 * λ * tr(ε(F))^2
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
    ε(F) = 0.5(F + F') -I3
    ∂Ψuu(F) = μ * (δᵢₖδⱼₗ3D + δᵢₗδⱼₖ3D) + λ * δᵢⱼδₖₗ3D
    ∂Ψu(F) = ∂Ψuu(F) ⊙ (F - I3)
    Ψ(F) = μ * sum(ε(F).*ε(F)) + 0.5 * λ * tr(ε(F))^2
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct VolumetricEnergy{A} <: Elasto
  λ::Float64
  Kinematic::A
  function VolumetricEnergy(; λ::Float64, Kinematic::KinematicModel=Kinematics(Elasto))
    new{typeof(Kinematic)}(λ, Kinematic)
  end

  function (obj::VolumetricEnergy)(Λ::Float64=1.0)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    λ = obj.λ
    Ψ(F) = (λ / 2.0) * (J(F) - 1)^2 
    ∂Ψ_∂J(F) =  λ * (J(F) - 1)
    ∂Ψ2_∂J2(F) = λ
    ∂Ψu(F) = ∂Ψ_∂J(F) * H(F)
    ∂Ψuu(F) = ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ×ᵢ⁴(∂Ψ_∂J(F) * F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct NeoHookean3D{A} <: Elasto
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
    ∂Ψ2_∂J2(F) = -μ * ∂log2∂J2(J(F)) + λ
    ∂Ψuu(F) = μ * I9 + ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂J(F) * ×ᵢ⁴(F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct MooneyRivlin3D{A} <: Mechano
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

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂H(F) × F + ∂Ψ_∂J(F) * H(F)
    ∂Ψuu(F) = μ1 * I9 + μ2 * (F × (I9 × F)) + ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ×ᵢ⁴(∂Ψ_∂H(F) + ∂Ψ_∂J(F) * F)

    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct MooneyRivlin2D{A} <: Mechano
  λ::Float64
  μ1::Float64
  μ2::Float64
  ρ::Float64
  Kinematic::A

  function MooneyRivlin2D(; λ::Float64, μ1::Float64, μ2::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ1, μ2, ρ, Kinematic)
  end

  function (obj::MooneyRivlin2D)(Λ::Float64=1.0; Threshold=0.01)
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


struct NonlinearMooneyRivlin3D{A} <: Mechano
  λ::Float64
  μ1::Float64
  μ2::Float64
  α::Float64
  β::Float64
  ρ::Float64
  Kinematic::A
  function NonlinearMooneyRivlin3D(; λ::Float64, μ1::Float64, μ2::Float64, α::Float64, β::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ1, μ2, α, β, ρ, Kinematic)
  end

  function (obj::NonlinearMooneyRivlin3D)(Λ::Float64=1.0; Threshold=0.01)
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
    ∂ΨFF(F) = (2 * μ1 * (α - 1) / (3.0^(α - 1)) * (tr((F)' * F))^(α - 2)) * (F ⊗ F) + (μ1 / (3.0^(α - 1)) * (tr((F)' * F))^(α - 1)) * I9
    ∂ΨHH(F) = (2 * μ2 * (β - 1) / (3.0^(β - 1)) * (tr((H(F))' * H(F)))^(β - 2)) * (H(F) ⊗ H(F)) + (μ2 / (3.0^(β - 1)) * (tr((H(F))' * H(F)))^(β - 1)) * I9
    ∂Ψuu(F) = ∂ΨFF(F) + (F × (∂ΨHH(F) × F)) + ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ×ᵢ⁴(∂Ψ_∂H(F) + ∂Ψ_∂J(F) * F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct NonlinearMooneyRivlin2D{A} <: Mechano
  λ::Float64
  μ1::Float64
  μ2::Float64
  α::Float64
  β::Float64
  ρ::Float64
  Kinematic::A
  function NonlinearMooneyRivlin2D(; λ::Float64, μ1::Float64, μ2::Float64, α::Float64, β::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ1, μ2, α, β, ρ, Kinematic)
  end

  function (obj::NonlinearMooneyRivlin2D)(Λ::Float64=1.0; Threshold=0.01)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    λ, μ1, μ2, α, β = obj.λ, obj.μ1, obj.μ2, obj.α, obj.β

    Ψ(F) = μ1 / (2.0 * α * 3.0^(α - 1)) * (tr((F)' * F) + 1.0)^α + μ2 / (2.0 * β * 3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^β - (μ1 + 2.0 * μ2) * logreg(J(F)) +
           (λ / 2.0) * (J(F) - 1)^2

    ∂Ψ_∂F(F) = ((μ1 / (3.0^(α - 1)) * (tr((F)' * F) + 1.0)^(α - 1)) + μ2 / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 1)) * F
    ∂log∂J(J) = J >= Threshold ? 1 / J : (2 / Threshold - J / (Threshold^2))
    ∂log2∂J2(J) = J >= Threshold ? -1 / (J^2) : (-1 / (Threshold^2))
    ∂Ψ_∂J(F) = μ2 / (3.0^(β - 1)) * J(F) * (tr((F)' * F) + J(F)^2)^(β - 1) - (μ1 + 2.0 * μ2) * ∂log∂J(J(F)) + λ * (J(F) - 1)

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)

    ∂Ψ2_∂FF(F) = ((μ1 / (3.0^(α - 1)) * (tr((F)' * F) + 1.0)^(α - 1)) + μ2 / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 1)) * I4 +
                 2 * ((μ1 * (α - 1) / (3.0^(α - 1)) * (tr((F)' * F) + 1.0)^(α - 2)) + μ2 * (β - 1) / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 2)) * (F ⊗ F)
    ∂Ψ2_∂FJ(F) = (2 * μ2 * (β - 1) / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 2)) * J(F) * F
    ∂Ψ2_∂JJ(F) = μ2 / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 1) + (2 * μ2 * (β - 1) / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 2)) * J(F)^2 - (μ1 + 2.0 * μ2) * ∂log2∂J2(J(F)) + λ

    ∂Ψuu(F) = ∂Ψ2_∂FF(F) + (∂Ψ2_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ2_∂FJ(F)) + ∂Ψ2_∂JJ(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂J(F) * _∂H∂F_2D()
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct NonlinearMooneyRivlin2D_CV{A} <: Mechano
  λ::Float64
  μ1::Float64
  μ2::Float64
  α::Float64
  β::Float64
  γ::Float64
  ρ::Float64
  Kinematic::A
  function NonlinearMooneyRivlin2D_CV(; λ::Float64, μ1::Float64, μ2::Float64, α::Float64, β::Float64, γ::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ1, μ2, α, β, γ, ρ, Kinematic)
  end

  function (obj::NonlinearMooneyRivlin2D_CV)(Λ::Float64=1.0)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    λ, μ1, μ2, α, β, γ = obj.λ, obj.μ1, obj.μ2, obj.α, obj.β, obj.γ

    Ψ(F) = μ1 / (2.0 * α * 3.0^(α - 1)) * (tr((F)' * F) + 1.0)^α + μ2 / (2.0 * β * 3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^β - (μ1 + 2.0 * μ2) * log(J(F)) +
           (λ) * (J(F)^(γ) + J(F)^(-γ))

    ∂Ψ_∂F(F) = ((μ1 / (3.0^(α - 1)) * (tr((F)' * F) + 1.0)^(α - 1)) + μ2 / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 1)) * F
    ∂Ψ_∂J(F) = μ2 / (3.0^(β - 1)) * J(F) * (tr((F)' * F) + J(F)^2)^(β - 1) - (μ1 + 2.0 * μ2) * (1.0 / J(F)) + λ * γ * (J(F)^(γ - 1) - J(F)^(-γ - 1))

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)

    ∂Ψ2_∂FF(F) = ((μ1 / (3.0^(α - 1)) * (tr((F)' * F) + 1.0)^(α - 1)) + μ2 / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 1)) * I4 +
                 2 * ((μ1 * (α - 1) / (3.0^(α - 1)) * (tr((F)' * F) + 1.0)^(α - 2)) + μ2 * (β - 1) / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 2)) * (F ⊗ F)
    ∂Ψ2_∂FJ(F) = (2 * μ2 * (β - 1) / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 2)) * J(F) * F
    ∂Ψ2_∂JJ(F) = μ2 / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 1) + (2 * μ2 * (β - 1) / (3.0^(β - 1)) * (tr((F)' * F) + J(F)^2)^(β - 2)) * J(F)^2 + (μ1 + 2.0 * μ2) * (1.0 / (J(F))^2) + λ * γ * ((γ - 1) * J(F)^(γ - 2) + (γ + 1) * J(F)^(-γ - 2))

    ∂Ψuu(F) = ∂Ψ2_∂FF(F) + (∂Ψ2_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ2_∂FJ(F)) + ∂Ψ2_∂JJ(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂J(F) * _∂H∂F_2D()
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct NonlinearMooneyRivlin_CV{A} <: Mechano
  λ::Float64
  μ1::Float64
  μ2::Float64
  α::Float64
  β::Float64
  γ::Float64
  ρ::Float64
  Kinematic::A
  function NonlinearMooneyRivlin_CV(; λ::Float64, μ1::Float64, μ2::Float64, α::Float64, β::Float64, γ::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ1, μ2, α, β, γ, ρ, Kinematic)
  end

  function (obj::NonlinearMooneyRivlin_CV)(Λ::Float64=1.0)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    λ, μ1, μ2, α, β, γ = obj.λ, obj.μ1, obj.μ2, obj.α, obj.β, obj.γ

    Ψ(F) = μ1 / (2.0 * α * 3.0^(α - 1)) * (tr((F)' * F))^α + 
           μ2 / (2.0 * β * 3.0^(β - 1)) * (tr((H(F))' * H(F)))^β - 
           (μ1 + 2*μ2) * log(J(F)) + λ * (J(F)^(γ) + J(F)^(-γ))

    ∂Ψ_∂F(F) = ((μ1 / (3.0^(α - 1)) * (trAA(F))^(α - 1))) * F
    ∂Ψ_∂H(F) = ((μ2 / (3.0^(β - 1)) * (tr((H(F))' * H(F)))^(β - 1))) * H(F)
    ∂Ψ_∂J(F) = -(μ1 + 2*μ2) * (1.0 / J(F)) + λ * γ * (J(F)^(γ - 1) - J(F)^(-γ - 1))
    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂H(F) × F + ∂Ψ_∂J(F) * H(F)

    ∂Ψ2_∂FF(F) = ((μ1 / (3.0^(α - 1)) * (tr((F)' * F))^(α - 1))) * I9 +
                 2 * ((μ1 * (α - 1) / (3.0^(α - 1)) * (tr((F)' * F))^(α - 2))) * (F ⊗ F)
    ∂Ψ2_∂HH(F) = ((μ2 / (3.0^(β - 1)) * (tr((H(F))' * H(F)))^(β - 1))) * I9 +
                 2 * ((μ2 * (β - 1) / (3.0^(β - 1)) * (tr((H(F))' * H(F)))^(β - 2))) * (H(F) ⊗ H(F))
    ∂Ψ2_∂JJ(F) = (μ1 + 2*μ2) * (1.0 / (J(F))^2) + λ * γ * ((γ - 1) * J(F)^(γ - 2) + (γ + 1) * J(F)^(-γ - 2))

    ∂Ψuu(F) = ∂Ψ2_∂FF(F) + (F × (∂Ψ2_∂HH(F) × F)) + ∂Ψ2_∂JJ(F) * (H(F) ⊗ H(F)) + ×ᵢ⁴(∂Ψ_∂H(F) + ∂Ψ_∂J(F) * F)
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
    λ, μ, α, γ = obj.λ, obj.μ, obj.α, obj.γ

    Ψ(F) = μ / (2.0 * α * 3.0^(α - 1)) * (tr((F)' * F))^α - μ * log(J(F)) + λ * (J(F)^(γ) + J(F)^(-γ))

    ∂Ψ_∂F(F) = ((μ / (3.0^(α - 1)) * (tr((F)' * F))^(α - 1))) * F
    ∂Ψ_∂J(F) = -μ * (1.0 / J(F)) + λ * γ * (J(F)^(γ - 1) - J(F)^(-γ - 1))

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)

    ∂Ψ2_∂FF(F) = ((μ / (3.0^(α - 1)) * (tr((F)' * F))^(α - 1))) * I9 +
                 2 * ((μ * (α - 1) / (3.0^(α - 1)) * (tr((F)' * F))^(α - 2))) * (F ⊗ F)
    ∂Ψ2_∂JJ(F) = μ * (1.0 / (J(F))^2) + λ * γ * ((γ - 1) * J(F)^(γ - 2) + (γ + 1) * J(F)^(-γ - 2))

    ∂Ψuu(F) = ∂Ψ2_∂FF(F) + ∂Ψ2_∂JJ(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂J(F) * ×ᵢ⁴(F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct NonlinearIncompressibleMooneyRivlin2D_CV{A} <: Mechano
  λ::Float64
  μ::Float64
  α::Float64
  γ::Float64
  ρ::Float64
  Kinematic::A
  function NonlinearIncompressibleMooneyRivlin2D_CV(; λ::Float64, μ::Float64, α::Float64, γ::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(λ, μ, α, γ, ρ, Kinematic)
  end

  function (obj::NonlinearIncompressibleMooneyRivlin2D_CV)(Λ::Float64=1.0)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    λ, μ, α, γ = obj.λ, obj.μ, obj.α, obj.γ

    e(F) = (tr((F)' * F) + 1.0) * J(F)^(-2 / 3)
    ∂e_∂F(F) = 2 * J(F)^(-2 / 3) * F
    ∂e_∂J(F) = -(2 / 3) * (tr((F)' * F) + 1.0) * J(F)^(-5 / 3)
    ∂e2_∂F2(F) = 2 * J(F)^(-2 / 3) * I4
    ∂e2_∂J2(F) = (10 / 9) * J(F)^(-8 / 3) * (tr((F)' * F) + 1.0)
    ∂e2_∂FJ(F) = -(4 / 3) * J(F)^(-5 / 3) * F

    Ψ1(F) = μ / (2 * α) * (e(F))^α
    Ψ2(F) = (λ) * (J(F)^(γ) + J(F)^(-γ))
    Ψ(F) = Ψ1(F) + Ψ2(F)

    ∂Ψ1_∂F(F) = (μ / 2) * (((e(F))^(α - 1.0)) * ∂e_∂F(F))
    ∂Ψ1_∂J(F) = (μ / 2) * (((e(F))^(α - 1.0)) * ∂e_∂J(F))
    ∂Ψ2_∂J(F) = λ * γ * (J(F)^(γ - 1) - J(F)^(-γ - 1))
    ∂Ψ_∂F(F) = ∂Ψ1_∂F(F)
    ∂Ψ_∂J(F) = ∂Ψ1_∂J(F) + ∂Ψ2_∂J(F)
    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)

    ∂Ψ1_∂F2(F) = (μ / 2) * ((e(F)^(α - 1)) * ∂e2_∂F2(F) + (α - 1) * (e(F)^(α - 2)) * ∂e_∂F(F) ⊗ ∂e_∂F(F))
    ∂Ψ1_∂J2(F) = (μ / 2) * ((e(F)^(α - 1)) * ∂e2_∂J2(F) + (α - 1) * (e(F)^(α - 2)) * ∂e_∂J(F) * ∂e_∂J(F))
    ∂Ψ1_∂FJ(F) = (μ / 2) * ((e(F)^(α - 1)) * ∂e2_∂FJ(F) + (α - 1) * (e(F)^(α - 2)) * ∂e_∂F(F) * ∂e_∂J(F))
    ∂Ψ2_∂J2(F) = λ * γ * ((γ - 1) * J(F)^(γ - 2) + (γ + 1) * J(F)^(-γ - 2))

    ∂Ψ_∂F2(F) = ∂Ψ1_∂F2(F)
    ∂Ψ_∂FJ(F) = ∂Ψ1_∂FJ(F)
    ∂Ψ_∂J2(F) = ∂Ψ1_∂J2(F) + ∂Ψ2_∂J2(F)

    ∂Ψuu(F) = ∂Ψ_∂F2(F) + ∂Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ_∂FJ(F) + ∂Ψ_∂J(F) * _∂H∂F_2D()
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct EightChain{A} <: Elasto
  μ::Float64
  N::Float64
  Kinematic::A
  function EightChain(; μ::Float64, N::Float64, Kinematic::KinematicModel=Kinematics(Elasto))
    new{typeof(Kinematic)}(μ,N,Kinematic)
  end

  function (obj::EightChain)(Λ::Float64=1.0)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    μ, N = obj.μ, obj.N

    Ψ(F) = begin
      C = F' * F
      C_iso = J(F)^(-2/3) * C
      β = sqrt(tr(C_iso) / 3 / N)
      L = β * (3.0 - β^2) / (1.0 - β^2)
      μ * N * (β * L + log(L / sinh(L)))
    end

    ∂Ψ∂F(F) = begin
      C = F' * F
      C_iso = J(F)^(-2 / 3) * C
      β = sqrt(tr(C_iso) / 3 / N)
      L = β * (3.0 - β^2) / (1.0 - β^2)
      ∂β∂I1_ = 0.5 / sqrt(tr(C_iso) * 3 * N)
      ∂L∂I1_ = ((3 * (1 - β^2)^2 + 2 * β * (3 * β - β^3)) / (1 - β^2)^2) * ∂β∂I1_
      n = (∂L∂I1_ * sinh(L) - L * cosh(L) * ∂L∂I1_)
      d = (L * sinh(L))
      ∂Ψ∂I1_ = μ * N * (∂β∂I1_ * L + β * ∂L∂I1_ + n / d)
      ∂I1_∂F = 2 * J(F)^(-2 / 3) * F
      ∂I1_∂J = -(2 / 3) * J(F)^(-5 / 3) * tr(C)
      ∂Ψ∂I1_ * (∂I1_∂F + ∂I1_∂J * H(F))
    end

    ∂Ψ∂FF(F) = begin
      H_ = H(F)
      C = F' * F
      C_iso = det(F)^(-2 / 3) * C
      β = sqrt(tr(C_iso) / 3 / N)
      L = β * (3.0 - β^2) / (1.0 - β^2)
      ∂β∂I1_ = 0.5 / sqrt(tr(C_iso) * 3 * N)
      ∂L∂I1_ = ((3 * (1 - β^2)^2 + 2 * β * (3 * β - β^3)) / (1 - β^2)^2) * ∂β∂I1_
      ∂β∂I1I1_ = -(3 * N) / (4 * (3 * N * tr(C_iso))^(3 / 2))
      ∂L∂I1I1_ = ((4 * β * (β^2 + 3)) / (1 - β^2)^3) * ∂β∂I1_^2 + ((3 * (1 - β^2)^2 + 2 * β * (3 * β - β^3)) / (1 - β^2)^2) * ∂β∂I1I1_
      ∂I1_∂F = 2 * det(F)^(-2 / 3) * F
      ∂I1_∂J = -(2 / 3) * det(F)^(-5 / 3) * tr(C)
      ∂I1_∂F∂F = 2 * det(F)^(-2 / 3) * I9
      ∂I1_∂J∂J = (10 / 9) * det(F)^(-8 / 3) * tr(C)
      ∂I1_∂F∂J = -(4 / 3) * det(F)^(-5 / 3) * F
      n = (∂L∂I1_ * sinh(L) - L * cosh(L) * ∂L∂I1_)
      d = (L * sinh(L))
      ∂n∂I1_ = ∂L∂I1I1_ * sinh(L) + ∂L∂I1_ * ∂L∂I1_* cosh(L) - ∂L∂I1_^2 * cosh(L) - L * sinh(L) * ∂L∂I1_^2 - L * cosh(L) * ∂L∂I1I1_
      ∂d∂I1_ = ∂L∂I1_ * sinh(L) + L * ∂L∂I1_* cosh(L)
      ∂Ψ∂I1_ = μ * N * (∂β∂I1_ * L + β * ∂L∂I1_ + n / d)
      ∂Ψ∂I1I1_ = μ * N * (∂β∂I1I1_ * L + 2 * ∂β∂I1_ * ∂L∂I1_ + β * ∂L∂I1I1_ + (∂n∂I1_ * d - n * ∂d∂I1_) / d^2)
      ∂Ψ∂I1I1_ * ((∂I1_∂F + ∂I1_∂J * H_) ⊗ (∂I1_∂F + ∂I1_∂J * H_)) + ∂Ψ∂I1_ * (∂I1_∂F∂F + ∂I1_∂F∂J ⊗ H_ + H_ ⊗ ∂I1_∂F∂J + ∂I1_∂J∂J * (H_ ⊗ H_) + I9 × (∂I1_∂J * F))
    end
    return (Ψ, ∂Ψ∂F, ∂Ψ∂FF)
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

    ∂ΨFF(F, N) = μ * (I4(F, N)^(α - 1)) * (I3 ⊗₁₃²⁴ (N ⊗ N)) + 2μ * (α - 1) * I4(F, N)^(α - 2) * (((F * N) ⊗ N) ⊗ ((F * N) ⊗ N))
    ∂ΨHH(F, N) = μ * (I5(F, N)^(β - 1)) * (I3 ⊗₁₃²⁴ (N ⊗ N)) + 2μ * (β - 1) * I5(F, N)^(β - 2) * (((H(F) * N) ⊗ N) ⊗ ((H(F) * N) ⊗ N))
    ∂Ψuu(F, N) = ∂ΨFF(F, N) + (F × (∂ΨHH(F, N) × F)) + ∂Ψ2_∂J2(F, N) * (H(F) ⊗ H(F)) + ×ᵢ⁴(∂Ψ_∂H(F, N) + ∂Ψ_∂J(F, N) * F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct TransverseIsotropy2D{A} <: Mechano
  μ::Float64
  α::Float64
  β::Float64
  ρ::Float64
  Kinematic::A
  function TransverseIsotropy2D(; μ::Float64, α::Float64, β::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(μ, α, β, ρ, Kinematic)
  end

  function (obj::TransverseIsotropy2D)(Λ::Float64=1.0; Threshold=0.01)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    I4(F, N) = (F * N) ⋅ (F * N)
    I5(F, N) = (H(F) * N) ⋅ (H(F) * N)
    μ, α, β = obj.μ, obj.α, obj.β
    Ψ(F, N) = μ / (2.0 * α) * (I4(F, N)^α - 1) + μ / (2.0 * β) * (I5(F, N)^β - 1) - μ * logreg(J(F))

    ∂I4∂F(F, N) = 2*((F * N) ⊗ N)
    ∂I4∂F∂F(F, N) = 2*(I2 ⊗₁₃²⁴ (N ⊗ N))
    ∂I5∂F∂F(F, N) =  2*(I2 ⊗ I2) -  2*(I2 ⊗ (N ⊗ N) + (N ⊗ N) ⊗ I2) +  2*((N ⊗ N) ⊗₁₃²⁴ I2)
    ∂I5∂F(F, N) = 2*tr(F)*I2 - 2*(N⋅(F*N))*I2 - 2*tr(F)*(N ⊗ N) +  2*(N ⊗ (F'*N))

    ∂log∂J(J) = J >= Threshold ? 1 / J : (2 / Threshold - J / (Threshold^2))
    ∂log2∂J2(J) = J >= Threshold ? -1 / (J^2) : (-1 / (Threshold^2))
    ∂Ψ_∂J(F, N) = -μ * ∂log∂J(J(F))
    ∂Ψ2_∂J2(F, N) = -μ * ∂log2∂J2(J(F))

    ∂Ψu(F, N) = (μ/2 * (I4(F, N)^(α - 1))) * ∂I4∂F(F, N) + 
                      (μ/2 * (I5(F, N)^(β - 1))) * ∂I5∂F(F, N)  + 
                      ∂Ψ_∂J(F, N) * H(F)

    ∂Ψuu(F, N) = μ/2*(α-1)*(I4(F, N)^(α - 2)) * (∂I4∂F(F, N)) ⊗ (∂I4∂F(F, N)) + 
                 μ/2*(I4(F, N)^(α - 1)) * ∂I4∂F∂F(F, N) + 
                 μ/2*(β-1)*(I5(F, N)^(β - 2)) * (∂I5∂F(F, N)) ⊗ (∂I5∂F(F, N)) + 
                 μ/2*(I5(F, N)^(β - 1)) * ∂I5∂F∂F(F, N) + 
                 ∂Ψ2_∂J2(F, N) * (H(F) ⊗ H(F)) + 
                 ∂Ψ_∂J(F, N)*_∂H∂F_2D()
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct IncompressibleNeoHookean3D{A} <: Elasto
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

    ∂Ψ1_∂J2(F) = (5 / 9) * μ * J(F)^(-8 / 3) * (tr((F)' * F))
    ∂Ψ2_∂J2(F) = λ
    ∂Ψ3_∂J2(F) = β / J_(F)^2
    ∂Ψ_∂J2(F) = (∂Ψ1_∂J2(F) * ∂J(F)^2 + ∂Ψ1_∂J(F) * ∂2J(F)) + ∂Ψ2_∂J2(F) + ∂Ψ3_∂J2(F)
    ∂Ψ_∂FJ(F) = -(2 / 3) * μ * J(F)^(-5 / 3) * ∂J(F) * F

    ∂Ψuu(F) = μ * I9 * J(F)^(-2 / 3) + ∂Ψ2_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ_∂FJ(F) + ∂Ψ_∂J(F) * ×ᵢ⁴(F)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end

  function (obj::IncompressibleNeoHookean3D)(::KinematicDescription{:SecondPiola}, Λ::Float64=1.0)
    Ψ(C) = obj.μ / 2 * tr(C) * det(C)^(-1/3)
    S(C) = begin
      detC = det(C)
      invC = inv(C)
      obj.μ * detC^(-1/3) * I3 - obj.μ / 3 * tr(C) * detC^(-1/3) * invC
    end
    ∂S∂C(C) = begin
      detC = det(C)
      trC = tr(C)
      invC = inv(C)
      IinvC = I3 ⊗ invC
      1/3 * obj.μ * detC^(-1/3) * (4/3*trC*invC⊗invC -(IinvC+IinvC') -trC/detC*×ᵢ⁴(C))
    end
    return (Ψ, S, ∂S∂C)
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

    ∂Ψ1_∂J2(F) = (5 / 9) * μ * J(F)^(-8 / 3) * (tr((F)' * F) + 1.0)
    ∂Ψ2_∂J2(F) = λ
    ∂Ψ3_∂J2(F) = β / J_(F)^2
    ∂Ψ_∂J2(F) = (∂Ψ1_∂J2(F) * ∂J(F)^2 + ∂Ψ1_∂J(F) * ∂2J(F)) + ∂Ψ2_∂J2(F) + ∂Ψ3_∂J2(F)
    ∂Ψ_∂FJ(F) = -(2 / 3) * μ * J(F)^(-5 / 3) * ∂J(F) * F
    ∂Ψuu(F) = μ * I4 * J(F)^(-2 / 3) + ∂Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ_∂FJ(F) + ∂Ψ_∂J(F) * _∂H∂F_2D()
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

    ∂Ψ1_∂J2(F) = (5 / 9) * μ * J(F)^(-8 / 3) * (tr((F)' * F) + 1.0)
    ∂Ψ2_∂J2(F) = λ * γ * ((γ - 1) * J(F)^(γ - 2) + (γ + 1) * J(F)^(-γ - 2))
    ∂Ψ_∂J2(F) = ∂Ψ1_∂J2(F) + ∂Ψ2_∂J2(F)
    ∂Ψ_∂FJ(F) = -(2 / 3) * μ * J(F)^(-5 / 3) * F
    ∂Ψuu(F) = μ * I4 * J(F)^(-2 / 3) + ∂Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂Ψ_∂FJ(F) + ∂Ψ_∂J(F) * _∂H∂F_2D()
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

    ∂Ψ1_∂J2(F) = μ * J(F)^(-3) * (tr((F)' * F))
    ∂Ψ2_∂J2(F) = β / J_(F)^2
    ∂Ψ_∂J2(F) = (∂Ψ1_∂J2(F) * ∂J(F)^2 + ∂Ψ1_∂J(F) * ∂2J(F)) + ∂Ψ2_∂J2(F)
    ∂Ψ_∂FJ(F) = -μ * J(F)^(-2) * F * ∂J(F)
    ∂Ψ_∂FF(F) = μ * J(F)^(-1) * I4

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

    Ψ(F) = μ * 0.5 * J(F)^(-1) * (tr((F)' * F))
    ∂Ψ_∂F(F) = μ * F * J(F)^(-1)
    ∂Ψ_∂J(F) = -μ / 2 * (tr((F)' * F)) * J(F)^(-2)

    ∂2Ψ_∂J2(F) = μ * J(F)^(-3) * (tr((F)' * F))
    ∂2Ψ_∂FJ(F) = -μ * J(F)^(-2) * F
    ∂2Ψ_∂FF(F) = μ * J(F)^(-1) * I4

    ∂Ψu(F) = ∂Ψ_∂F(F) + ∂Ψ_∂J(F) * H(F)
    ∂Ψuu(F) = ∂2Ψ_∂FF(F) + ∂2Ψ_∂J2(F) * (H(F) ⊗ H(F)) + ∂2Ψ_∂FJ(F) ⊗ H(F) + H(F) ⊗ ∂2Ψ_∂FJ(F) + ∂Ψ_∂J(F) * _∂H∂F_2D()

    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


struct IncompressibleNeoHookean3D_2dP{A} <: Mechano
  μ::Float64
  τ::Float64
  Δt::Float64
  ρ::Float64
  Kinematic::A

  function IncompressibleNeoHookean3D_2dP(; μ::Float64, τ::Float64, Δt::Float64, ρ::Float64=0.0, Kinematic::KinematicModel=Kinematics(Mechano))
    new{typeof(Kinematic)}(μ, τ, Δt, ρ, Kinematic)
  end

  function (obj::IncompressibleNeoHookean3D_2dP)(Λ::Float64=1.0; Threshold=0.01)
    _, H, J = get_Kinematics(obj.Kinematic; Λ=Λ)
    μ = obj.μ

    Ψ(Ce) = μ / 2 * tr(Ce) * (det(Ce))^(-1 / 3)
    ∂Ψ∂Ce(Ce) = μ / 2 * I3 * (det(Ce))^(-1 / 3)
    ∂Ψ∂dCe(Ce) = -μ / 6 * tr(Ce) * (det(Ce))^(-4 / 3)
    Se(Ce) = let HCe=H(Ce); 2 * (∂Ψ∂Ce(Ce) + ∂Ψ∂dCe(Ce) * HCe) end
    ∂2Ψ∂CedCe(Ce) = -μ / 6 * I3 * (det(Ce))^(-4 / 3)
    ∂2Ψ∂2dCe(Ce) = 2 * μ / 9 * tr(Ce) * (det(Ce))^(-7 / 3)
    ∂Se∂Ce(Ce) = let HCe=H(Ce); 2 * (∂2Ψ∂2dCe(Ce) * (HCe ⊗ HCe) + ∂2Ψ∂CedCe(Ce) ⊗ HCe + HCe ⊗ ∂2Ψ∂CedCe(Ce) + ∂Ψ∂dCe(Ce) * ×ᵢ⁴(Ce)) end
    
    return (Ψ, Se, ∂Se∂Ce)
  end
end
