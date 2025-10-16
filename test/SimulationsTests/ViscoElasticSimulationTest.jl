
BASE_FOLDER = dirname(dirname(pathof(HyperFEM)))
filename = joinpath(BASE_FOLDER, "test/data/ViscoElasticSimulation.jl")
include(filename)

λx, σΓ = visco_elastic_simulation()

if Int(VERSION.minor) > 10
  @test σΓ[end] ≈ 13870.815
else
  @test ≈(σΓ[end], 13870.815, rtol=1e-6)
end
