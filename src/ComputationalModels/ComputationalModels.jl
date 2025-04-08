module ComputationalModels
using DrWatson
using HyperFEM.PhysicalModels

using Gridap
using Gridap.Helpers
using Gridap.TensorValues, Gridap.FESpaces, Gridap.Algebra
using Gridap: createvtk
using Gridap.MultiField
using Gridap.FESpaces: get_assembly_strategy


using BlockArrays

using GridapPETSc, GridapPETSc.PETSC
using GridapPETSc: PetscScalar, PetscInt, PETSC,  @check_error_code

using GridapDistributed
using GridapDistributed: DistributedDiscreteModel, DistributedTriangulation,
  DistributedFESpace, DistributedDomainContribution, to_parray_of_arrays,
  allocate_in_domain, DistributedCellField, DistributedMultiFieldCellField,
  DistributedMultiFieldFEBasis, BlockPMatrix, BlockPVector, change_ghost

using PartitionedArrays
using PartitionedArrays: getany, tuple_of_arrays, matching_ghost_indices

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.NonlinearSolvers, GridapSolvers.BlockSolvers
using GridapSolvers.SolverInterfaces: SolverVerboseLevel, SOLVER_VERBOSE_NONE, SOLVER_VERBOSE_LOW, SOLVER_VERBOSE_HIGH
# using GridapSolvers.SolverInterfaces: init!, update!, finalize!
using GridapSolvers.SolverInterfaces: finished, print_message, converged

using LinearAlgebra
using WriteVTK

using GridapGmsh 
using GridapGmsh: GmshDiscreteModel
using GridapGmsh:@check_if_loaded
# using Gmsh: Gmsh, gmsh


include("BoundaryConditions.jl")
export DirichletBC
export NeumannBC
export get_Neumann_dΓ
export NothingBC
export NothingTC
export MultiFieldBC
export SingleFieldTC
export MultiFieldTC
export residual_Neumann
export updateBC!

include("FESpaces.jl")
export TrialFESpace
export TrialFESpace!
export TestFESpace
export TestFESpace!

include("GridapExtras.jl")
export GmshDiscreteModel
 
include("Drivers.jl")
export StaggeredModel
export StaticNonlinearModel
export DynamicNonlinearModel
export StaticLinearModel
export solve!
# export evaluate!
export project_dirichlet!
export get_state
export get_measure
export get_spaces
export get_assemblers
export get_trial_space
export get_test_space

include("PostProcessors.jl")
export PostProcessor
export Cauchy
export Entropy
export D0
end
