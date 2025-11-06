using Gridap
using Gridap.FESpaces
using GridapSolvers
using GridapSolvers.NonlinearSolvers
using HyperFEM
using HyperFEM.ComputationalModels.CartesianTags
using HyperFEM.ComputationalModels:constant
using HyperFEM.ComputationalModels:triangular
using HyperFEM.ComputationalModels.PostMetrics

function visco_elastic_simulation(;t_end=15, writevtk=true, verbose=true)
  # Domain and tessellation
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
  μv₁ = 3.153e5  # Pa
  τv₁ = 10.72    # s
  hyper_elastic_model = NeoHookean3D(λ=λ, μ=μ)
  viscous_branch = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μv₁), τ=τv₁)
  cons_model = GeneralizedMaxwell(hyper_elastic_model, viscous_branch)

  # Dirichlet boundary conditions
  strain = 0.5
  D_bc = DirichletBC(
    ["corner1", "corner2", "fixed", "moving"],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [long * strain, 0.0, 0.0]],
    [constant(), constant(), constant(), triangular(10/t_end)])
  dirichlet_masks = [
    [true, true, true], [true, false, true], [true, false, false], [true, false, false]]

  # Setup integration
  order = 2
  degree = 2 * order
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  dΓ = get_Neumann_dΓ(model, NothingBC(), degree)
  Γ1  = BoundaryTriangulation(model, tags=D_bc.tags[4])
  dΓ1 = Measure(Γ1, degree)
  Δt = 0.05

  # FE spaces
  reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
  Vu = TestFESpace(model, reffe, D_bc, dirichlet_masks=dirichlet_masks, conformity=:H1)
  VL2 = FESpace(Ω, reffe, conformity=:L2)
  Uu = TrialFESpace(Vu, D_bc, 0.0)
  Uun = TrialFESpace(Vu, D_bc, 0.0)

  uh = FEFunction(Uu, zero_free_values(Uu))
  unh = FEFunction(Uun, zero_free_values(Uun))
  state_vars = initializeStateVariables(cons_model, dΩ)

  k=Kinematics(Mechano,Solid)
  res(Λ) = (u,v)->residual(cons_model, k, u, v, dΩ, t_end * Λ, Δt, unh, state_vars)
  jac(Λ) = (u,du,v)->jacobian(cons_model, k, u, du, v, dΩ, t_end * Λ, Δt, unh, state_vars)

  ls = LUSolver()
  nls = NewtonSolver(ls; maxiter=20, atol=1.e-6, rtol=1.e-6, verbose=verbose)
  comp_model = StaticNonlinearModel(res, jac, Uu, Vu, D_bc; nls=nls, xh=uh, xh⁻=unh)

  λx = Float64[]
  σΓ = Float64[]
  F,_,_ = get_Kinematics(k)

  function driverpost(post)
    σh11, _... = Cauchy(cons_model, Kinematics(Mechano,Solid),uh, unh, state_vars, Ω, dΩ, 0.0, Δt)
    σΓ1 = sum(∫(σh11)dΓ1) / sum(∫(1.0)dΓ1)
    push!(σΓ, σΓ1)
    push!(λx, 1.0 + component_LInf(uh, :x, Ω) / long)
    updateStateVariables!(state_vars, cons_model, Δt, F∘(∇(uh)'), F∘(∇(unh)'))
  end

  post_model = PostProcessor(comp_model, driverpost; is_vtk=writevtk, filepath="")
  solve!(comp_model; stepping=(nsteps=Int(t_end/Δt), maxbisec=1), post=post_model, ProjectDirichlet=true)
  (λx, σΓ)
end


if abspath(PROGRAM_FILE) == @__FILE__
  using Plots
  λx, σΓ = visco_elastic_simulation()
  plot(λx, σΓ)
end
