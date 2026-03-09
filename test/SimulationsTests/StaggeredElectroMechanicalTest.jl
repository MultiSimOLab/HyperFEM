using HyperFEM
using Test

filename = projdir("test/data/StaggeredElectroMechanicalSimulation.jl")
include(filename)

φh, uh = staggered_electro_mechanical_simulation(is_vtk=false, verbose=false)

@test norm(uh[1]) ≈ 0.01492384116
@test norm(φh[1]) ≈ 0.00023052926
