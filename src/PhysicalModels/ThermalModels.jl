
# ===================
# Thermal models
# ===================

struct ThermalModel <: Thermo
  Cv::Real
  θr::Real
  α::Real
  κ::Real
  function ThermalModel(; Cv::Real, θr::Real, α::Real, κ::Real=10.0)
    new(Cv, θr, α, κ)
  end
end

function (obj::ThermalModel)(Λ::Float64=1.0)
  Ψ(δθ) = obj.Cv * (δθ - (δθ + obj.θr) * log((δθ + obj.θr) / obj.θr))
  ∂Ψθ(δθ) = -obj.Cv * log((δθ + obj.θr) / obj.θr)
  ∂Ψθθ(δθ) = -obj.Cv / (δθ + obj.θr)
  return (Ψ, ∂Ψθ, ∂Ψθθ)
end


# ===================
# Thermal laws
# ===================

struct EntropicElasticityLaw <: ThermalLaw
  θr::Real
  γ::Real
  EntropicElasticityLaw(; θr, γ) = new(θr, γ)
end

function (law::EntropicElasticityLaw)()
  @unpack θr, γ = law
  f(θ) = (θ/θr)^(γ+1)
  ∂f(θ) = (γ+1) * θ^γ / θr^(γ+1)
  ∂∂f(θ) = γ*(γ+1) * θ^(γ-1) / θr^(γ+1)
  return (f, ∂f, ∂∂f)
end

struct NonlinearMeltingLaw <: ThermalLaw
  θr::Real
  θM::Real
  γ::Real
  NonlinearMeltingLaw(; θr, θM, γ) = new(θr, θM, γ)
end

function (law::NonlinearMeltingLaw)()
  @unpack θr, θM, γ = law
  f(θ) = (1 - (θ/θM)^(γ+1)) / (1 - (θr/θM)^(γ+1))
  ∂f(θ) = -(γ+1)*θ^γ/θM^(γ+1) / (1 - (θr/θM)^(γ+1))
  ∂∂f(θ) = -γ*(γ+1)*θ^(γ-1)/θM^(γ+1) / (1 - (θr/θM)^(γ+1))
  return (f, ∂f, ∂∂f)
end

struct NonlinearSofteningLaw <: ThermalLaw
  θr::Real
  θt::Real
  γ::Real
  δ::Real
  NonlinearSofteningLaw(; θr, θt, γ, δ=0) = new(θr, θt, γ, δ)
end

function (law::NonlinearSofteningLaw)()
  @unpack θr, θt, γ, δ = law
  u(θ) = exp(-(θ/θt)^(γ+1))
  C = (1-δ) * u(θr) + δ
  f(θ) = ((1-δ) * u(θ) + δ) / C
  ∂f(θ) = -(1-δ)/C * (γ+1)/θt * (θ/θt)^γ * u(θ)
  ∂∂f(θ) = (1-δ)/C * (γ+1)/θ^2 * (θ/θt)^(γ+1) * ((γ+1)*(θ/θt)^(γ+1)-γ) * u(θ)
  return (f, ∂f, ∂∂f)
end

struct TrigonometricLaw <: ThermalLaw
  θr::Real
  θM::Real
end

function (law::TrigonometricLaw)()
  @unpack θr, θM = law  
  g(θ) = θ/θr * sin(2π*θ/θM)
  G(θ) = 1/2/π * θM/θr * (1 - cos(2π*θ/θM))
  H(θ) = 1/2/π * θM/θr * (θ - θM/2/π * sin(2π*θ/θM))
  f(θ) = (H(θr) - H(θ)) / (H(θM) - H(θr)) + 1.0
  ∂f(θ) = -G(θ) / (H(θM) - H(θr))
  ∂∂f(θ) = -g(θ) / θ / (H(θM) - H(θr))
  return (f, ∂f, ∂∂f)
end

struct PolynomialLaw <: ThermalLaw
  θr::Real
  a::Real
  b::Real
  c::Real
end

function (law::PolynomialLaw)()
  @unpack θr, a, b, c = law
  f(θ)   = a*((θ-θr)/θr)^3  + b*((θ-θr)/θr)^2 + c*(θ-θr)/θr + 1
  ∂f(θ)  = 3a*(θ-θr)^2/θr^3 + 2b*(θ-θr)/θr^2 + c/θr
  ∂∂f(θ) = 6a*(θ-θr)/θr^3 + 2b/θr^2
  return (f, ∂f, ∂∂f)
end
