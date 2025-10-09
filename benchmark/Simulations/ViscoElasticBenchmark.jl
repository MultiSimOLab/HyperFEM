using Gridap
using Gridap.FESpaces
using GridapSolvers
using GridapSolvers.NonlinearSolvers
using HyperFEM
using HyperFEM.ComputationalModels.CartesianTags
using HyperFEM.ComputationalModels:constant
using HyperFEM.ComputationalModels:triangular

# Domain and Tessellation
long   = 0.05   # m
width  = 0.005  # m
thick  = 0.001  # m
domain = (0.0, long, 0.0, width, 0.0, thick)
partition = (3, 1, 1)
model = CartesianDiscreteModel(domain, partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels, "corner1", CartesianTags.corner000)
add_tag_from_tags!(labels, "corner2", CartesianTags.corner010)
add_tag_from_tags!(labels, "fixed", CartesianTags.faceX0)
add_tag_from_tags!(labels, "moving", CartesianTags.faceX1)

# Constitutive model
μ = 1.367e4    # Pa
λ = 1000μ      # Pa
hyper_elastic_model = NeoHookean3D(λ=λ, μ=μ)
μv₁ = 3.153e5  # Pa
τv₁ = 10.72    # s
viscous_branch = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μv₁), τv₁)
cons_model = GeneralizedMaxwell(hyper_elastic_model, viscous_branch)

# Setup integration
order = 2
degree = 2 * order
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)
dΓ = get_Neumann_dΓ(model, NothingBC(), degree)
Δt = 0.05
t_end = 5

# Dirichlet conditions 
strain = 1.5
D_bc = DirichletBC(
  ["corner1", "corner2", "fixed", "moving"],
  [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [long * strain, 0.0, 0.0]],
  [constant(), constant(), constant(), triangular(10/t_end)])
dirichlet_masks = [
  [true, true, true], [true, false, true], [true, false, false], [true, false, false]]

# FE spaces
reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
Vu = TestFESpace(model, reffe, D_bc, dirichlet_masks=dirichlet_masks, conformity=:H1)
Uu = TrialFESpace(Vu, D_bc, 0.0)
Uun = TrialFESpace(Vu, D_bc, 0.0)

# residual and jacobian
uh = FEFunction(Uu, zero_free_values(Uu))
unh = FEFunction(Uu, zero_free_values(Uu))
state_vars = initializeStateVariables(cons_model, dΩ)

res(Λ) = (u,v)->residual(cons_model, u, v, dΩ, t_end * Λ, Δt, unh, state_vars)
jac(Λ) = (u,du,v)->jacobian(cons_model, u, du, v, dΩ, t_end * Λ, Δt, unh, state_vars)

ls = LUSolver()
nls_ = NewtonSolver(ls; maxiter=20, atol=1.e-6, rtol=1.e-6, verbose=true)
comp_model = StaticNonlinearModel(res, jac, Uu, Vu, D_bc; nls=nls_, xh=uh, xh⁻=unh)

function driverpost(post; cons_model=cons_model, Δt=Δt, uh=uh, unh=unh, A=state_vars, Ω=Ω, dΩ=dΩ)
  updateStateVariables!(cons_model, Δt, uh, unh, A)
end
post_model = PostProcessor(comp_model, driverpost; is_vtk=true, filepath=simdir)

# Solve
SUITE["Simulations"]["ViscoElastic"] = @benchmarkable solve!(comp_model; stepping=(nsteps=Int(t_end / Δt), maxbisec=1), post=post_model, ProjectDirichlet=true)
