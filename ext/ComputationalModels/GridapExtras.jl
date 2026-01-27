
import HyperFEM.ComputationalModels.get_local_matrix_type
import HyperFEM.ComputationalModels.get_local_vector_type
import HyperFEM.ComputationalModels.get_local_assembly_strategy

function get_local_matrix_type(a::GridapDistributed.DistributedSparseMatrixAssembler)
  return getany(map(get_matrix_type,a.assems))
end
function get_local_vector_type(a::GridapDistributed.DistributedSparseMatrixAssembler)
  return getany(map(get_vector_type,a.assems))
end
function get_local_assembly_strategy(a::GridapDistributed.DistributedSparseMatrixAssembler)
  return get_assembly_strategy(a)
end
