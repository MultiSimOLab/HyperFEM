
struct ElectroMechModel{E<:Electro,M<:Mechano} <: ElectroMechano{E,M}
  electro::E
  mechano::M

  function ElectroMechModel(electro::E, mechano::M) where {E<:Electro,M<:Mechano}
    new{E,M}(electro, mechano)
  end

  function ElectroMechModel(; electro::E, mechano::M) where {E<:Electro,M<:Mechano}
    new{E,M}(electro, mechano)
  end

  function (obj::ElectroMechModel{<:Electro,<:IsoElastic})(Λ::Float64=1.0)
    Ψm, ∂Ψm_u, ∂Ψm_uu = obj.mechano()
    Ψem, ∂Ψem_u, ∂Ψem_φ, ∂Ψem_uu, ∂Ψem_φu, ∂Ψem_φφ = _getCoupling(obj.electro, obj.mechano)
    Ψ(F, E) = Ψm(F) + Ψem(F, E)
    ∂Ψu(F, E) = ∂Ψm_u(F) + ∂Ψem_u(F, E)
    ∂Ψφ(F, E) = ∂Ψem_φ(F, E)
    ∂Ψuu(F, E) = ∂Ψm_uu(F) + ∂Ψem_uu(F, E)
    ∂Ψφu(F, E) = ∂Ψem_φu(F, E)
    ∂Ψφφ(F, E) = ∂Ψem_φφ(F, E)
    return (Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ)
  end
  function (obj::ElectroMechModel{<:Electro,<:AnisoElastic})(Λ::Float64=1.0)
    Ψm, ∂Ψm_u, ∂Ψm_uu = obj.mechano()
    Ψem, ∂Ψem_u, ∂Ψem_φ, ∂Ψem_uu, ∂Ψem_φu, ∂Ψem_φφ = _getCoupling(obj.electro, obj.mechano)
    Ψ(F, E, N) = Ψm(F, N) + Ψem(F, E)
    ∂Ψu(F, E, N) = ∂Ψm_u(F, N) + ∂Ψem_u(F, E)
    ∂Ψφ(F, E, N) = ∂Ψem_φ(F, E)
    ∂Ψuu(F, E, N) = ∂Ψm_uu(F, N) + ∂Ψem_uu(F, E)
    ∂Ψφu(F, E, N) = ∂Ψem_φu(F, E)
    ∂Ψφφ(F, E, N) = ∂Ψem_φφ(F, E)
    return (Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ)
  end
  function (obj::ElectroMechModel{<:Electro,<:ViscoElastic{<:IsoElastic}})(Λ::Float64=1.0)
    Ψm, ∂Ψm_u, ∂Ψm_uu = obj.mechano()
    Ψem, ∂Ψem_u, ∂Ψem_φ, ∂Ψem_uu, ∂Ψem_φu, ∂Ψem_φφ = _getCoupling(obj.electro, obj.mechano)
    Ψ(F, E, Fn, A...) = Ψm(F, Fn, A...) + Ψem(F, E)
    ∂Ψu(F, E, Fn, A...) = ∂Ψm_u(F, Fn, A...) + ∂Ψem_u(F, E)
    ∂Ψφ(F, E, Fn, A...) = ∂Ψem_φ(F, E)
    ∂Ψuu(F, E, Fn, A...) = ∂Ψm_uu(F, Fn, A...) + ∂Ψem_uu(F, E)
    ∂Ψφu(F, E, Fn, A...) = ∂Ψem_φu(F, E)
    ∂Ψφφ(F, E, Fn, A...) = ∂Ψem_φφ(F, E)
    return (Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ)
  end
  function (obj::ElectroMechModel{<:Electro,<:ViscoElastic{<:AnisoElastic}})(Λ::Float64=1.0)
    Ψm, ∂Ψm_u, ∂Ψm_uu = obj.mechano()
    Ψem, ∂Ψem_u, ∂Ψem_φ, ∂Ψem_uu, ∂Ψem_φu, ∂Ψem_φφ = _getCoupling(obj.electro, obj.mechano)
    Ψ(F, E, n, Fn, A...) = Ψm(F, n, Fn, A...) + Ψem(F, E)
    ∂Ψu(F, E, n, Fn, A...) = ∂Ψm_u(F, n, Fn, A...) + ∂Ψem_u(F, E)
    ∂Ψφ(F, E, n, Fn, A...) = ∂Ψem_φ(F, E)
    ∂Ψuu(F, E, n, Fn, A...) = ∂Ψm_uu(F, n, Fn, A...) + ∂Ψem_uu(F, E)
    ∂Ψφu(F, E, n, Fn, A...) = ∂Ψem_φu(F, E)
    ∂Ψφφ(F, E, n, Fn, A...) = ∂Ψem_φφ(F, E)
    return (Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ)
  end
end

function update_time_step!(obj::ElectroMechModel, Δt::Float64)
  update_time_step!(obj.electro, Δt)
  update_time_step!(obj.mechano, Δt)
end

function initialize_state(obj::ElectroMechModel, points::Measure)
  initialize_state(obj.mechano, points)
end

function update_state!(obj::ElectroMechModel, state, F, E, args...)
  update_state!(obj.mechano, state, F, args...)
end

function Dissipation(obj::ElectroMechModel)
  Dvis = Dissipation(obj.mechano)
  D(F, E, X...) = Dvis(F, X...)
end


function _getCoupling(elec::Electro, mec::Mechano, Λ::Float64=0.0)
  J(F) = det(F)
  H(F) = det(F) * inv(F)'
  # Energy #
  HE(F, E) = H(F) * E
  HEHE(F, E) = HE(F, E) ⋅ HE(F, E)
  Ψem(F, E) = (-elec.ε / (2.0 * J(F))) * HEHE(F, E)
  # First Derivatives #
  ∂Ψem_∂H(F, E) = (-elec.ε / (J(F))) * (HE(F, E) ⊗ E)
  ∂Ψem_∂J(F, E) = (+elec.ε / (2.0 * J(F)^2.0)) * HEHE(F, E)
  ∂Ψem_∂E(F, E) = (-elec.ε / (J(F))) * (H(F)' * HE(F, E))
  ∂Ψem_u(F, E) = ∂Ψem_∂H(F, E) × F + ∂Ψem_∂J(F, E) * H(F)
  ∂Ψem_φ(F, E) = ∂Ψem_∂E(F, E)

  # Second Derivatives #
  ∂Ψem_HH(F, E) = (-elec.ε / (J(F))) * (I3 ⊗₁₃²⁴ (E ⊗ E))
  ∂Ψem_HJ(F, E) = (+elec.ε / (J(F))^2.0) * (HE(F, E) ⊗ E)
  ∂Ψem_JJ(F, E) = (-elec.ε / (J(F))^3.0) * HEHE(F, E)
  ∂Ψem_uu(F, E) = (F × (∂Ψem_HH(F, E) × F)) +
                  H(F) ⊗₁₂³⁴ (∂Ψem_HJ(F, E) × F) +
                  (∂Ψem_HJ(F, E) × F) ⊗₁₂³⁴ H(F) +
                  ∂Ψem_JJ(F, E) * (H(F) ⊗₁₂³⁴ H(F)) +
                  ×ᵢ⁴(∂Ψem_∂H(F, E) + ∂Ψem_∂J(F, E) * F)

  ∂Ψem_EH(F, E) = (-elec.ε / (J(F))) * ((I3 ⊗₁₃² HE(F, E)) + (H(F)' ⊗₁₂³ E))
  ∂Ψem_EJ(F, E) = (+elec.ε / (J(F))^2.0) * (H(F)' * HE(F, E))
  ∂Ψem_φu(F, E) = (∂Ψem_EH(F, E) × F) + (∂Ψem_EJ(F, E) ⊗₁²³ H(F))

  ∂Ψem_φφ(F, E) = (-elec.ε / (J(F))) * (H(F)' * H(F))

  return (Ψem, ∂Ψem_u, ∂Ψem_φ, ∂Ψem_uu, ∂Ψem_φu, ∂Ψem_φφ)
end


function (+)(Model1::Electro, Model2::Mechano)
  ElectroMechModel(Model1, Model2)
end
function (+)(Model1::Mechano, Model2::Electro)
  ElectroMechModel(Model2, Model1)
end

struct FlexoElectroModel{EM<:ElectroMechano} <: FlexoElectro{EM}
  electromechano::EM
  κ::Float64

  function FlexoElectroModel(electro::E, mechano::M; κ=1.0) where {E<:Electro,M<:Mechano}
    physmodel = ElectroMechModel(electro, mechano)
    new{ElectroMechModel{E,M}}(physmodel, κ)
  end

  function FlexoElectroModel(; electro::E, mechano::M, κ=1.0) where {E<:Electro,M<:Mechano}
    physmodel = ElectroMechModel(electro, mechano)
    new{ElectroMechModel{E,M}}(physmodel, κ)
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

    Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = obj.electromechano(Λ)
    return Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ, Φ
  end
end
