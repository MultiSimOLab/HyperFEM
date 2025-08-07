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



@time begin
    include("TensorAlgebra.jl")
end
@time begin
    include("PhysicalModels.jl")
end
@time begin
    include("WeakForms.jl")
end

