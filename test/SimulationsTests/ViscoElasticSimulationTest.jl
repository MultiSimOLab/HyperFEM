
# using PackageName
# BASE_FOLDER = dirname(dirname(pathof(PackageName)))
# test_file = joinpath(BASE_FOLDER, "data", "file.txt")

include("../../examples/ViscoElasticSimulation.jl")

λx, σΓ = visco_elastic_simulation()

if Int(VERSION.minor) > 10
  @test σΓ[end] ≈ 13870.815
else
  @test ≈(σΓ[end], 13870.815, rtol=1e-6)
end
