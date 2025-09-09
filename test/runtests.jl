using HyperFEM
using Test
using Profile
using Gridap, GridapGmsh, GridapSolvers, DrWatson, TimerOutputs
using GridapSolvers.NonlinearSolvers
using Gridap.FESpaces
using Gridap.CellData
using Gridap.TensorValues
using HyperFEM: jacobian, IterativeSolver, solve!
using BenchmarkTools

import Base.isapprox

isapprox(A::MultiValue, B::MultiValue; kwargs...) = isapprox(get_array(A), get_array(B); kwargs...)


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

