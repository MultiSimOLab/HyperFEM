
BASE_FOLDER = dirname(dirname(pathof(HyperFEM)))
filename = joinpath(BASE_FOLDER, "test/data/ViscoElasticSimulation.jl")
include(filename)

λx, σΓ = visco_elastic_simulation(t_end=5, write_vtk=false, verbose=false)

@test σΓ[end] ≈ 152821.386
