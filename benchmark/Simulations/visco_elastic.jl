using Gridap
using Gridap.FESpaces
using GridapSolvers
using GridapSolvers.NonlinearSolvers
using HyperFEM
using HyperFEM.ComputationalModels.CartesianTags
using HyperFEM.ComputationalModels:constant
using HyperFEM.ComputationalModels:triangular


name(x) = string(typeof(x).name.wrapper)
name(x::HyperFEM.PhysicalModels.ComposedElasticModel) = name(x.Model1) * "+" * name(x.Model2)
name(x::GeneralizedMaxwell) = string(typeof(x).name.wrapper) * "(" * name(x.LongTerm) * "," * string(length(x.Branches)) * "Branch)"


# Domain and Tessellation
long = 0.05   # m
width = 0.005  # m
thick = 0.001  # m
domain = (0.0, long, 0.0, width, 0.0, thick)
partition = (3, 1, 1)
model = CartesianDiscreteModel(domain, partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels, "corner1", CartesianTags.corner000)
add_tag_from_tags!(labels, "corner2", CartesianTags.corner010)
add_tag_from_tags!(labels, "fixed", CartesianTags.faceX0)
add_tag_from_tags!(labels, "moving", CartesianTags.faceX1)


# Constitutive models
μ = 1.367e4  # Pa
N = 7.860e5  # -
λ = μ * 10     # Pa
# hyper_elastic_model = EightChain(μ=μ, N=N) + VolumetricEnergy(λ=λ)
hyper_elastic_model = NeoHookean3D(λ=λ, μ=μ)

μv₁ = 3.153e5  # Pa
τv₁ = 10.72    # s
branch1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μv₁), τv₁)

μv₂ = 5.639e5  # Pa
τv₂ = 0.82     # s
branch2 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μv₂), τv₂)

μv₃ = 1.981e5  # Pa
τv₃ = 498.8    # s
branch3 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μv₃), τv₃)

cons_model = GeneralizedMaxwell(hyper_elastic_model, branch1)#, branch2, branch3)

# Preparing output
simdir = datadir("sims", name(cons_model))
setupfolder(simdir, remove=".vtu")

# Setup integration
order = 2
degree = 2 * order
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)
dΓ = get_Neumann_dΓ(model, NothingBC(), degree)

# Dirichlet conditions 
strain = 1.5
D_bc = DirichletBC(
  ["corner1", "corner2", "fixed", "moving"],
  [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [long * strain, 0.0, 0.0]],
  [constant(), constant(), constant(), triangular(10/25)])
dirichlet_masks = [
  [true, true, true], [true, false, true], [true, false, false], [true, false, false]]

# FE spaces
reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
Vu = TestFESpace(model, reffe, D_bc, dirichlet_masks=dirichlet_masks, conformity=:H1)
Uu = TrialFESpace(Vu, D_bc, 0.0)
Uun = TrialFESpace(Vu, D_bc, 0.0)

# residual and jacobian function of time
uh = FEFunction(Uu, zero_free_values(Uu))
unh = FEFunction(Uu, zero_free_values(Uu))
state_vars = initializeStateVariables(cons_model, dΩ)
Δt = 0.05
t_end = 25

res(Λ) = (u,v)->residual(cons_model, u, v, dΩ, t_end * Λ, Δt, unh, state_vars)
jac(Λ) = (u,du,v)->jacobian(cons_model, u, du, v, dΩ, t_end * Λ, Δt, unh, state_vars)

ls = LUSolver()
nls_ = NewtonSolver(ls; maxiter=20, atol=1.e-6, rtol=1.e-6, verbose=true)

comp_model = StaticNonlinearModel(res, jac, Uu, Vu, D_bc; nls=nls_, xh=uh, xh⁻=unh)


function driverpost(post; cons_model=cons_model, Δt=Δt, uh=uh, unh=unh, sv=state_vars, Ω=Ω, dΩ=dΩ)
  Λ = post.Λ[end]
  Λ_ = post.iter
  σ11, σ12, σ13, σ22, σ23, σ33, p = Cauchy(cons_model, uh, unh, sv, Ω, dΩ, Λ * t_end, Δt)
  pvd = post.cachevtk[3]
  filePath = post.cachevtk[2]
  if post.cachevtk[1]
    pvd[Λ_] = createvtk(Ω, filePath * "/TIME_$Λ_" * ".vtu", cellfields=["u" => uh, "σ11" => σ11, "σ12" => σ12, "σ13" => σ13, "σ22" => σ22, "σ23" => σ23, "σ33" => σ33, "p" => p])
  end
  updateStateVariables!(cons_model, Δt, uh, unh, sv)
end
post_model = PostProcessor(comp_model, driverpost; is_vtk=true, filepath=simdir)

# Solve
x = solve!(comp_model; stepping=(nsteps=Int(t_end / Δt), maxbisec=1), post=post_model, ProjectDirichlet=true)
