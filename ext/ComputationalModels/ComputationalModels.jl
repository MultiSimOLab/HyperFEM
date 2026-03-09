module ComputationalModels

using HyperFEM
using Gridap, GridapDistributed, GridapPETSc
using PartitionedArrays
using PartitionedArrays: getany

include("FESpaces.jl")
include("GridapExtras.jl")

end
