
# ===================
# Thermal models
# ===================

struct ThermalModel <: Thermo
  Cv::Float64
  θr::Float64
  α::Float64
  κ::Float64
  γv::Float64
  γd::Float64
  function ThermalModel(; Cv::Float64, θr::Float64, α::Float64, κ::Float64=10.0, γv::Float64=1.0, γd::Float64=1.0)
    new(Cv, θr, α, κ, γv, γd)
  end

  function (obj::ThermalModel)(Λ::Float64=1.0)
    Ψ(δθ) = obj.Cv * (δθ - (δθ + obj.θr) * log((δθ + obj.θr) / obj.θr))
    ∂Ψθ(δθ) = -obj.Cv * log((δθ + obj.θr) / obj.θr)
    ∂Ψθθ(δθ) = -obj.Cv / (δθ + obj.θr)
    return (Ψ, ∂Ψθ, ∂Ψθθ)
  end
end


struct ThermalModel3rdLaw <: Thermo
  cv0::Float64
  θr::Float64
  α::Float64
  κ::Float64
  γv::Float64
  γd::Float64
  function ThermalModel3rdLaw(; cv0::Float64, θr::Float64, α::Float64, κ::Float64, γv::Float64, γd::Float64)
    new(cv0, θr, α, κ, γv, γd)
  end
end

function volumetric_law(model::ThermalModel3rdLaw)
  θr, γ = model.θr, model.γv
  g(θ) = 1/(γ+1) * ((θ/θr)^(γ+1) -1)
  ∂g(θ) = θ^γ / θr^(γ+1)
  ∂∂g(θ) = γ*θ^(γ-1) / θr^(γ+1)
  return (g, ∂g, ∂∂g)
end

function isochoric_law(model::ThermalModel3rdLaw)
  θr, γ = model.θr, model.γd
  g(θ) = (θ/θr)^(-γ)
  ∂g(θ) = -γ*θ^(-γ-1) * θr^γ
  ∂∂g(θ) = γ*(γ+1)*θ^(-γ-2) * θr^γ
  return (g, ∂g, ∂∂g)
end

function (obj::ThermalModel3rdLaw)()
  throw("The thermal model 3rd law is not callable. Please, define the energy in combination with other models.")
end
