using Gridap
using HyperFEM
using Test

@testset "SimulationsTests" begin
   
    include("ViscoElasticSimulationTest.jl")

    include("StaticMechanicalDirichletTest.jl")

end
