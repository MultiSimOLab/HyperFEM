
import HyperFEM.instantiate_caches

function instantiate_caches(x, nls::PETScNonlinearSolver, op::NonlinearOperator)
  return GridapPETSc._setup_cache(x, nls, op)
end
