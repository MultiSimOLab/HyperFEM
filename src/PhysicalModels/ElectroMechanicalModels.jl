
struct ElectroMechModel{M<:Mechano, E<:Electro} <: ElectroMechano
  mechano::M
  electro::E

  function ElectroMechModel(mechano::M, electro::E) where {M<:Mechano, E<:Electro}
    new{M,E}(mechano,electro)
  end

  function ElectroMechModel(;mechano::M, electro::E) where {M<:Mechano, E<:Electro}
    new{M,E}(mechano,electro)
  end

  function (obj::ElectroMechModel)(Λ::Float64=1.0)
    Ψm, ∂Ψm_u, ∂Ψm_uu = obj.mechano(Λ)
    Ψem, ∂Ψem_u, ∂Ψem_φ, ∂Ψem_uu, ∂Ψem_φu, ∂Ψem_φφ = _getCoupling(obj.mechano, obj.electro, Λ)
    Ψ(F, E) = Ψm(F) + Ψem(F, E)
    ∂Ψu(F, E) = ∂Ψm_u(F) + ∂Ψem_u(F, E)
    ∂Ψφ(F, E) = ∂Ψem_φ(F, E)
    ∂Ψuu(F, E) = ∂Ψm_uu(F) + ∂Ψem_uu(F, E)
    ∂Ψφu(F, E) = ∂Ψem_φu(F, E)
    ∂Ψφφ(F, E) = ∂Ψem_φφ(F, E)
    return (Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ)
  end
  
  function (obj::ElectroMechModel{<:ViscoElastic,<:Electro})(Λ::Float64=1.0; Δt)
    Ψm, ∂Ψm_u, ∂Ψm_uu = obj.mechano(Λ, Δt=Δt)
    Ψem, ∂Ψem_u, ∂Ψem_φ, ∂Ψem_uu, ∂Ψem_φu, ∂Ψem_φφ = _getCoupling(obj.mechano, obj.electro, Λ)
    Ψ(F, Fn, A, E) = Ψm(F, Fn, A) + Ψem(F, E)
    ∂Ψu(F, Fn, A, E) = ∂Ψm_u(F, Fn, A) + ∂Ψem_u(F, E)
    ∂Ψφ(F, Fn, A, E) = ∂Ψem_φ(F, E)
    ∂Ψuu(F, Fn, A, E) = ∂Ψm_uu(F, Fn, A) + ∂Ψem_uu(F, E)
    ∂Ψφu(F, Fn, A, E) = ∂Ψem_φu(F, E)
    ∂Ψφφ(F, Fn, A, E) = ∂Ψem_φφ(F, E)
    return (Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ)
  end
end

ViscoElectricModel = ElectroMechModel{<:ViscoElastic,<:Electro}

function initializeStateVariables(obj::ElectroMechano, points::Measure)
  A = initializeStateVariables(obj.mechano, points)
  B = initializeStateVariables(obj.electro, points)
  (A, B)
end

function updateStateVariables!(state, obj::ElectroMechano, args...)
  updateStateVariables!(state[1], obj.mechano, args)
  updateStateVariables!(state[2], obj.electro, args)
end


function _getCoupling(mec::Mechano, elec::Electro, Λ::Float64)
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
