using HyperFEM
using Test

filename = projdir("test/data/StaggeredViscoElectricSimulation.jl")
include(filename)

t, uz = staggered_visco_electric_simulation(t_end=.5, writevtk=false, verbose=false)

@test uz[end] ≈ 0.00388461154
@test norm(uz) ≈ 0.0062104368
