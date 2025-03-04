include("../src/Mimosa.jl")
using Gridap
using TimerOutputs
using LinearAlgebra
using Test
using BenchmarkTools


 modelLE = Mimosa.LinearElasticity3D(3.0,1.0)
Ψ, ∂Ψu, ∂Ψuu = modelLE(Mimosa.DerivativeStrategy{:analytic}())
 
∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
 
 
@code_warntype  (Ψ(∇u))
@code_warntype   norm(∂Ψu(∇u))
@code_warntype norm(∂Ψuu(∇u))
 



 




 
