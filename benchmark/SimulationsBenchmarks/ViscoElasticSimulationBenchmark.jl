
filename = projdir("test/data/ViscoElasticSimulation.jl")
include(filename)

SUITE["Simulations"]["ViscoElastic"] = @benchmarkable visco_elastic_simulation(t_end=2, writevtk=false, verbose=false)
