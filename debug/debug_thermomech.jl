include("../src/GridapMultiSimO.jl")
using Gridap
using TimerOutputs
using LinearAlgebra
using Test
using BenchmarkTools

struct ThermalModelb{A}  
  Cv::Float64
  θr::Float64
  α::Float64
  κ::Float64
  DΨ::A
  function ThermalModelb(; Cv::Float64, θr::Float64, α::Float64, κ::Float64=10.0)  
    Ψ(δθ) = Cv * (δθ - (δθ + θr) * log((δθ + θr) / θr))
    ∂Ψθ(δθ) = -Cv * log((δθ + θr) / θr)
    ∂Ψθθ(δθ) = -Cv / (δθ + θr)
    DΨ=(Ψ, ∂Ψθ, ∂Ψθθ)
    new{typeof(DΨ)}(Cv, θr, α, κ, DΨ)
  end
end

∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
 
@time begin
modelT = GridapMultiSimO.ThermalModel(Cv=1.0, θr= 1.0,α= 2.0)
Ψ, ∂Ψθ, ∂Ψθθ=modelT(GridapMultiSimO.DerivativeStrategy{:analytic}())
Ψ(34.0)
∂Ψθ(34.0)
∂Ψθθ(34.0)
end

@time begin
modelTT=ThermalModelb(; Cv=1.0, θr= 1.0,α= 2.0)  
modelTT.DΨ[1](34.0)
modelTT.DΨ[2](34.0)
modelTT.DΨ[3](34.0)
end





 function f(θ::Float64)
  return θ/1.0
 end
 df(θ::Float64)::Float64=1.0
modelTM = Mimosa.ThermoMech(modelT, modelMR, f, df)
 
Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ= modelTM(Mimosa.DerivativeStrategy{:analytic}())

θt = 3.4
∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
∇φ = VectorValue(1.0, 2.0, 3.0)
 
    (Ψ(∇u, θt))
   norm(∂Ψu(∇u, θt))
   norm(∂Ψθ(∇u, θt))
  norm(∂Ψuu(∇u, θt))
norm(∂Ψθθ(∇u, θt))
  norm(∂Ψuθ(∇u, θt))




 




 
