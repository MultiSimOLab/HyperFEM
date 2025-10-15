
# using PackageName
# BASE_FOLDER = dirname(dirname(pathof(PackageName)))
# test_file = joinpath(BASE_FOLDER, "data", "file.txt")

include("../../examples/ViscoElasticSimulation.jl")

λx, σΓ = visco_elastic_simulation()
@test σΓ[end] ≈ 13870.814984291746
