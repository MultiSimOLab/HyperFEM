include("../src/Mimosa.jl")
using Gridap
using TimerOutputs
using LinearAlgebra
using Test
using BenchmarkTools




 modelMR = Mimosa.MoneyRivlin3D(3.0, 1.0, 2.0)
 modelID = Mimosa.IdealDielectric(4.0)
 modelT = Mimosa.ThermalModel(1.0, 1.0, 2.0)
 Ψθ, ∂Ψθ, ∂Ψθθ = modelT(Mimosa.DerivativeStrategy{:analytic}())

 function f(θ::Float64)
  return θ/1.0
 end

 df(θ::Float64)::Float64=1.0
modelTEM = Mimosa.ThermoElectroMech(modelT, modelID, modelMR, f, df)

 


 
Ψ, ∂Ψu, ∂Ψφ, ∂Ψθ, ∂Ψuu, ∂Ψφφ, ∂Ψθθ, ∂Ψφu, ∂Ψuθ, ∂Ψφθ = modelTEM(Mimosa.DerivativeStrategy{:analytic}())
 
lu_(v)   = residual_TEM(::CouplingStrategy{:staggered_M}(),(uh,φh,θh),v, DΨ.∂Ψu, dΩ)



θt = 3.4
∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
∇φ = VectorValue(1.0, 2.0, 3.0)
 
    (Ψ(∇u,∇φ, θt))
   norm(∂Ψu(∇u,∇φ, θt))
   norm(∂Ψφ(∇u,∇φ, θt))
   norm(∂Ψθ(∇u,∇φ, θt))
   norm(∂Ψuu(∇u,∇φ, θt))
   norm(∂Ψφφ(∇u,∇φ, θt))
   norm(∂Ψθθ(∇u,∇φ, θt))
   norm(∂Ψφu(∇u,∇φ, θt))
   norm(∂Ψuθ(∇u,∇φ, θt))
   norm(∂Ψφθ(∇u,∇φ, θt))




 




 
