using HyperFEM.TensorAlgebra
using BenchmarkTools


function _δδ_μ_2D(μ::Float64)
  TensorValue{4,4,Float64,16}(
    2 * μ,
    0.0,
    0.0,
    0.0,
    0.0,
    μ,
    μ,
    0.0,
    0.0,
    μ,
    μ,
    0.0,
    0.0,
    0.0,
    0.0,
    2.0 * μ)
end

function _δδ_λ_2D(λ::Float64)
  TensorValue{4,4,Float64,16}(
    λ,
    0.0,
    0.0,
    λ,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    λ,
    0.0,
    0.0,
    λ)
end

SUITE["Tensor algebra"]["δδ_μ_2d"] = @benchmarkable δᵢₖδⱼₗ2D + δᵢₗδⱼₖ2D
SUITE["Tensor algebra"]["δδ_λ_2d"] = @benchmarkable 1.0 * δᵢⱼδₖₗ2D
