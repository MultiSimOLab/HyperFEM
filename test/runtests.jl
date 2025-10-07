using HyperFEM
using Test
using Gridap, GridapGmsh, GridapSolvers, DrWatson, TimerOutputs
using GridapSolvers.NonlinearSolvers
using Gridap.FESpaces
using Gridap.CellData
using Gridap.TensorValues
using HyperFEM: jacobian, IterativeSolver, solve!


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

end;

