using Gridap.TensorValues
using HyperFEM.PhysicalModels
using BenchmarkTools


function benchmark_viscous_model()
  elasto = NeoHookean3D(λ=1e6, μ=1e3)
  visco = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=1e3), 10.)
  model = GeneralizedMaxwell(elasto, visco)
  Ψ, ∂Ψu, ∂Ψuu = model(Δt = 1e-2)
  F = TensorValue(1.:9...) * 1e-3 + I3
  Fn = TensorValue(1.:9...) * 5e-4 + I3
  Uvn = TensorValue(1.,2.,3.,2.,4.,5.,3.,5.,6.) * 2e-4 + I3
  J = det(F)
  Uvn *= J^(-1/3)
  λvn = 1e-3
  Avn = VectorValue(Uvn.data..., λvn)
  SUITE["Constitutive models"]["Visco-elastic Ψ"] = @benchmarkable $Ψ($F, $Fn, $Avn)
  SUITE["Constitutive models"]["Visco-elastic ∂Ψu"] = @benchmarkable $∂Ψu($F, $Fn, $Avn)
  SUITE["Constitutive models"]["Visco-elastic ∂Ψuu"] = @benchmarkable $∂Ψuu($F, $Fn, $Avn)
end

benchmark_viscous_model()
