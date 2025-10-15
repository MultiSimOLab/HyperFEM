
# using PackageName
# BASE_FOLDER = dirname(dirname(pathof(PackageName)))
# test_file = joinpath(BASE_FOLDER, "data", "file.txt")

include("../../examples/ViscoElasticSimulation.jl")

SUITE["Simulations"]["ViscoElastic"] = @benchmarkable visco_elastic_simulation()
