
BASE_FOLDER = dirname(dirname(pathof(HyperFEM)))
filename = joinpath(BASE_FOLDER, "test/data/StaticMechanicalNeumannSimulation.jl")
include(filename)

x = static_mechanical_neumann_simulation(writevtk=false, verbose=false)

@test norm(x) ≈ 1.000148588846
