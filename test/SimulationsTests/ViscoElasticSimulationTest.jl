
BASE_FOLDER = dirname(dirname(pathof(HyperFEM)))
filename = joinpath(BASE_FOLDER, "test/data/ViscoElasticSimulation.jl")
include(filename)

λx, σΓ = visco_elastic_simulation(t_end=5, write_vtk=false, verbose=false)

@test σΓ[end] ≈ 152821.386

if Int(VERSION.minor) > 10
  @test σΓ[end] ≈ 152821.386
else
  @test ≈(σΓ[end], 152821.386, rtol=1e-6)
end
