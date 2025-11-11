
struct ThermoElectroMechModel{T<:Thermo,E<:Electro,M<:Mechano} <: ThermoElectroMechano
  thermo::T
  electro::E
  mechano::M
  fθ::Function
  dfdθ::Function
  
  function ThermoElectroMechModel(thermo::T, electro::E, mechano::M; fθ::Function, dfdθ::Function) where {T<:Thermo,E<:Electro,M<:Mechano}
    new{T,E,M}(thermo, electro, mechano, fθ, dfdθ)
  end

  function ThermoElectroMechModel(; thermo::T, electro::E, mechano::M, fθ::Function, dfdθ::Function) where {T<:Thermo,E<:Electro,M<:Mechano}
    new{T,E,M}(thermo, electro, mechano, fθ, dfdθ)
  end

  function (obj::ThermoElectroMechModel)(Λ::Float64=1.0)
    Ψt, ∂Ψt_θ, ∂Ψt_θθ = obj.thermo(Λ)
    Ψm, ∂Ψm_u, ∂Ψm_uu = obj.mechano(Λ)
    Ψem, ∂Ψem_u, ∂Ψem_φ, ∂Ψem_uu, ∂Ψem_φu, ∂Ψem_φφ = _getCoupling(obj.electro, obj.mechano, Λ)
    Ψtm, ∂Ψtm_u, ∂Ψtm_θ, ∂Ψtm_uu, ∂Ψtm_uθ, ∂Ψtm_θθ = _getCoupling(obj.thermo, obj.mechano, Λ)
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


struct ThermoElectroMech_Govindjee{T<:Thermo,E<:Electro,M<:Mechano} <: ThermoElectroMechano
  thermo::T
  electro::E
  mechano::M
  fθ::Function
  dfdθ::Function
  gθ::Function
  dgdθ::Function
  β::Float64

  function ThermoElectroMech_Govindjee(thermo::T, electro::E, mechano::M; fθ::Function, dfdθ::Function, gθ::Function, dgdθ::Function, β::Float64=0.0) where {T<:Thermo,E<:Electro,M<:Mechano}
    new{T,E,M}(thermo, electro, mechano, fθ, dfdθ, gθ, dgdθ, β)
  end

  function ThermoElectroMech_Govindjee(; thermo::T, electro::E, mechano::M, fθ::Function, dfdθ::Function, gθ::Function, dgdθ::Function, β::Float64=0.0) where {T<:Thermo,E<:Electro,M<:Mechano}
    new{T,E,M}(thermo, electro, mechano, fθ, dfdθ, gθ, dgdθ, β)
  end

  function (obj::ThermoElectroMech_Govindjee)(Λ::Float64=1.0)
    Ψm, _, _ = obj.mechano(Λ)
    Ψem, _, _, _, _, _ = _getCoupling(obj.electro, obj.mechano, Λ)
    f(δθ) = obj.fθ(δθ)
    df(δθ) = obj.dfdθ(δθ)
    g(δθ) = obj.gθ(δθ)
    dg(δθ) = obj.dgdθ(δθ)

    J(F) = det(F)
    H(F) = det(F) * inv(F)'
    Ψer(F) = obj.thermo.α * (J(F) - 1.0) * obj.thermo.θr
    ΨL1(δθ) = obj.thermo.Cv * obj.thermo.θr * (1 - obj.β) * ((δθ + obj.thermo.θr) / obj.thermo.θr * (1.0 - log((δθ + obj.thermo.θr) / obj.thermo.θr)) - 1.0)
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


struct ThermoElectroMech_Bonet{T<:Thermo,E<:Electro,M<:Mechano} <: ThermoElectroMechano
  thermo::T
  electro::E
  mechano::M
  
  function ThermoElectroMech_Bonet(thermo::T, electro::E, mechano::M) where {T<:Thermo,E<:Electro,M<:Mechano}
    new{T,E,M}(thermo, electro, mechano)
  end

  function ThermoElectroMech_Bonet(; thermo::T, electro::E, mechano::M) where {T<:Thermo,E<:Electro,M<:Mechano}
    new{T,E,M}(thermo, electro, mechano)
  end

  function (obj::ThermoElectroMech_Bonet)(Λ::Float64=1.0)
    @unpack Cv,θr, α, κ, γv, γd = obj.thermo
    Ψem, ∂Ψem∂F, ∂Ψem∂E, ∂Ψem∂FF, ∂Ψem∂EF, ∂Ψem∂EE = _getCoupling(obj.electro, obj.mechano, Λ)
    gd(δθ) = 1/(γd+1) * (((δθ+θr)/θr)^(γd+1) -1)
    ∂gd(δθ) = (δθ+θr)^γd / θr^(γd+1)
    ∂∂gd(δθ) = γd*(δθ+θr)^(γd-1) / θr^(γd+1)
    gv(δθ) = 1/(γv+1) * (((δθ+θr)/θr)^(γv+1) -1)
    ∂gv(δθ) = (δθ+θr)^γv / θr^(γv+1)
    ∂∂gv(δθ) = γv*(δθ+θr)^(γv-1) / θr^(γv+1)

    J(F) = det(F)
    H(F) = det(F) * inv(F)'

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

function Dissipation(obj::ThermoElectroMech_Bonet, Δt)
  @unpack Cv,θr, α, κ, γv, γd = obj.thermo
  Dvis = Dissipation(obj.mechano, Δt)
  gd(δθ) = 1/(γd+1) * (((δθ+θr)/θr)^(γd+1) -1)
  ∂gd(δθ) = (δθ+θr)^γd / θr^(γd+1)
  D(F, E, δθ, A...) = (1 + gd(δθ)) * Dvis(F, A...)
  ∂D∂θ(F, E, δθ, A...) = ∂gd(δθ) * Dvis(F, A...)
  return(D, ∂D∂θ)
end
