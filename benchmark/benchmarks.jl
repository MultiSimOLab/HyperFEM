using BenchmarkTools
using Gridap
using HyperFEM

const SUITE = BenchmarkGroup()

include("TensorAlgebraBenchmarks/benchmarks.jl")

include("ConstitutiveModelsBenchmark/benchmarks.jl")

include("SimulationsBenchmarks/benchmarks.jl")
