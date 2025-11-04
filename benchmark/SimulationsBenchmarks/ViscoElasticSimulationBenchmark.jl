
BASE_FOLDER = dirname(dirname(pathof(HyperFEM)))
filename = joinpath(BASE_FOLDER, "test/data/ViscoElasticSimulation.jl")
include(filename)

SUITE["Simulations"]["ViscoElastic"] = @benchmarkable visco_elastic_simulation(writevtk=false, verbose=false)
