
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
