
BASE_FOLDER = dirname(dirname(pathof(HyperFEM)))
filename = joinpath(BASE_FOLDER, "test/data/StaticMechanicalNeumannSimulation.jl")
include(filename)

SUITE["Simulations"]["StaticMechanicalNeumann"] = @benchmarkable static_mechanical_neumann_simulation(writevtk=false, verbose=false)
