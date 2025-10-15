
# ===================
# MultiPhysicalModel models
# ===================

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
