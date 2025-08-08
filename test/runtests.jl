using HyperFEM
using Test
using Profile
using Gridap, GridapGmsh, GridapSolvers, DrWatson, TimerOutputs
using GridapSolvers.NonlinearSolvers
using Gridap.FESpaces
using Gridap.CellData
using HyperFEM: jacobian, IterativeSolver, solve!
using WriteVTK
using Revise
using DrWatson
using JSON
using ForwardDiff
using BenchmarkTools


@testset "HyperFEMTests" verbose = true begin

    @time begin
        include("../test/TestConstitutiveModels/runtests.jl")
    end
    @time begin
        include("./TestTensorAlgebra/runtests.jl")
    end
    @time begin
        include("../test/TestWeakForms/runtests.jl")
    end

end

