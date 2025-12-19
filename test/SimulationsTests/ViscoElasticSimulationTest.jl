using HyperFEM
using Test

filename = projdir("test/data/ViscoElasticSimulation.jl")
include(filename)

λx, σΓ = visco_elastic_simulation(t_end=2, writevtk=false, verbose=false)

@test σΓ[end] ≈ 21872.5028
