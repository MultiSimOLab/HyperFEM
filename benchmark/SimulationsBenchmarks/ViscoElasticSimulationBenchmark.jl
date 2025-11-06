
filename = projdir("test/data/ViscoElasticSimulation.jl")
include(filename)

SUITE["Simulations"]["ViscoElastic"] = @benchmarkable visco_elastic_simulation(writevtk=false, verbose=false)
