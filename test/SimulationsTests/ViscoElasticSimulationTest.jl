
filename = projdir("test/data/ViscoElasticSimulation.jl")
include(filename)

λx, σΓ = visco_elastic_simulation(t_end=5, writevtk=false, verbose=false)

@test σΓ[end] ≈ 152821.386
