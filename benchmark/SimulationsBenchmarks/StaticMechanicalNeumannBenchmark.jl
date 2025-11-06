
filename = projdir("test/data/StaticMechanicalNeumannSimulation.jl")
include(filename)

SUITE["Simulations"]["StaticMechanicalNeumann"] = @benchmarkable static_mechanical_neumann_simulation(writevtk=false, verbose=false)
