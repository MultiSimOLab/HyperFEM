
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
  Cv0::Float64
  θr::Float64
  α::Float64
  κ::Float64
  γv::Float64
  γd::Float64
  function ThermalModel3rdLaw(; Cv0::Float64, θr::Float64, α::Float64, κ::Float64, γv::Float64, γd::Float64)
    new(Cv0, θr, α, κ, γv, γd)
  end
end

function (obj::ThermalModel3rdLaw)()
  @unpack Cv0, θr, α, κ, γv, γd = obj
  g(θ,θr,γ) = 1/(γ+1) * ((θ/θr)^(γ+1) -1)
  ∂g(θ,θr,γ) = θ^γ / θr^(γ+1)
  ∂∂g(θ,θr,γ) = γ*θ^(γ-1) / θr^(γ+1)
  gd(θ) = g(θ,θr,γd)
  ∂gd(θ) = ∂g(θ,θr,γd)
  ∂∂gd(θ) = ∂∂g(θ,θr,γd)
  gv(θ) = g(θ,θr,γv)
  ∂gv(θ) = ∂g(θ,θr,γv)
  ∂∂gv(θ) = ∂∂g(θ,θr,γv)
  return (gv, ∂gv, ∂∂gv, gd, ∂gd, ∂∂gd)
end
