
SUITE["Simulations"] = BenchmarkGroup()

include("ViscoElasticSimulationBenchmark.jl")

include("StaticMechanicalDirichletBenchmark.jl")

include("StaticMechanicalNeumannBenchmark.jl")
