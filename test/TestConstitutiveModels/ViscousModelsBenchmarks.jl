using Gridap.TensorValues
using HyperFEM.PhysicalModels
using BenchmarkTools


function benchmark_viscous_model(model)
  Ψ, ∂Ψu, ∂Ψuu = model(Δt = 1e-2)
  F = TensorValue(1.:9...) * 1e-3 + I3
  Fn = TensorValue(1.:9...) * 5e-4 + I3
  Uvn = TensorValue(1.,2.,3.,2.,4.,5.,3.,5.,6.) * 2e-4 + I3
  J = det(F)
  Uvn *= J^(-1/3)
  λvn = 1e-3
  Avn = VectorValue(Uvn.data..., λvn)
  print("Ψ(F, Fn, Avn)    |")
  @btime $Ψ($F, $Fn, $Avn)
  print("∂Ψu(F, Fn, Avn)  |")
  @btime $∂Ψu($F, $Fn, $Avn)
  print("∂Ψuu(F, Fn, Avn) |")
  @btime $∂Ψuu($F, $Fn, $Avn)
end


elasto = NeoHookean3D(λ=λ, μ=μ)
visco = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μ1), τ1)
visco_elastic = GeneralizedMaxwell(elasto, visco)
benchmark_viscous_model(visco);
benchmark_viscous_model(visco_elastic);
