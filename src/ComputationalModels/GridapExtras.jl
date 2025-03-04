

function GridapGmsh.GmshDiscreteModel(mshfile; terminal=1, renumber=true)

  @check_if_loaded
  if !isfile(mshfile)
    error("Msh file not found: $mshfile")
  end

  gmsh.initialize()
  gmsh.option.setNumber("General.Terminal", terminal)
  gmsh.option.setNumber("Mesh.SaveAll", 1)
  gmsh.option.setNumber("Mesh.MedImportGroupsOfNodes", 1)
  gmsh.open(mshfile)
  renumber && gmsh.model.mesh.renumberNodes()
  renumber && gmsh.model.mesh.renumberElements()
  model = GmshDiscreteModel(gmsh)
  gmsh.finalize()
  model
end


function repeat_spaces(nblocks::Integer,U0::FESpace,V0::FESpace)
  U = MultiFieldFESpace([U0 for i in 1:nblocks];style=BlockMultiFieldStyle())
  V = MultiFieldFESpace([V0 for i in 1:nblocks];style=BlockMultiFieldStyle())
  return U,V
end

function repeated_allocate_in_domain(nblocks::Integer,M::AbstractMatrix)
  mortar(map(i -> allocate_in_domain(M), 1:nblocks))
end


get_local_matrix_type(a::Assembler) = get_matrix_type(a)
get_local_vector_type(a::Assembler) = get_vector_type(a)
get_local_assembly_strategy(a::Assembler) = get_assembly_strategy(a)

function get_local_matrix_type(a::GridapDistributed.DistributedSparseMatrixAssembler)
  return getany(map(get_matrix_type,a.assems))
end
function get_local_vector_type(a::GridapDistributed.DistributedSparseMatrixAssembler)
  return getany(map(get_vector_type,a.assems))
end
function get_local_assembly_strategy(a::GridapDistributed.DistributedSparseMatrixAssembler)
  return get_assembly_strategy(a)
end

function get_local_matrix_type(a::MultiField.BlockSparseMatrixAssembler)
  return get_local_matrix_type(first(a.block_assemblers))
end
function get_local_vector_type(a::MultiField.BlockSparseMatrixAssembler)
  return get_local_vector_type(first(a.block_assemblers))
end
function get_local_assembly_strategy(a::MultiField.BlockSparseMatrixAssembler)
  return get_local_assembly_strategy(first(a.block_assemblers))
end

