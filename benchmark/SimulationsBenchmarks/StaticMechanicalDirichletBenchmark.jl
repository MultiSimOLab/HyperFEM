
BASE_FOLDER = dirname(dirname(pathof(HyperFEM)))
filename = joinpath(BASE_FOLDER, "test/data/StaticMechanicalDirichletSimulation.jl")
include(filename)

SUITE["Simulations"]["StaticMechanicalDirichlet"] = @benchmarkable static_mechanical_dirichlet_simulation(writevtk=false, verbose=false)
