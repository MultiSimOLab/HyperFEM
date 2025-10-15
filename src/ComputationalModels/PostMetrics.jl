module PostMetrics

using Gridap
using HyperFEM.PhysicalModels

"""
    component_LInf(::FEFunction, ::Symbol, ::Triangulation)::Float64

Calculate the L-inf norm of a vector-valued finite element function.
It could be useful to find the maximum displacement.

# Example
    x_max = component_LInf(uh, :x, Ω)
"""
function component_LInf(u, dir, Ω)
  if     dir === :x
    n = VectorValue(1.0, 0.0, 0.0)
  elseif dir === :y
    n = VectorValue(0.0, 1.0, 0.0)
  elseif dir === :z
    n = VectorValue(0.0, 0.0, 1.0)
  else
    throw("Direction must be either :x, :y or :z. Got $dir")
  end
  reffe = ReferenceFE(lagrangian, Float64, 1)
  V = FESpace(Ω, reffe, conformity=:L2)
  un = interpolate_everywhere(u⋅n, V)
  uall = [un.free_values; un.dirichlet_values]
  norm(uall, Inf)
end


"""
Calculate the variation of the volume with respect to the undeformed configuration.
"""
function volume_diff(uh, dΩ)
  F, _, J = Kinematics(Mechano).metrics
  sum(∫(J ∘ F ∘ ∇(uh) -1.0)dΩ) / sum(∫(1.0)dΩ)
end


end