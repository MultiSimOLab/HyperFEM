
BASE_FOLDER = dirname(dirname(pathof(HyperFEM)))
filename = joinpath(BASE_FOLDER, "test/data/StaticMechanicalDirichletSimulation.jl")
include(filename)

x = static_mechanical_dirichlet_simulation(writevtk=false, verbose=false)

@test norm(x) ≈ 0.27148722276
