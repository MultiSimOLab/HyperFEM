using Gridap, GridapGmsh, GridapMultiSimO, GridapSolvers, DrWatson, TimerOutputs
using GridapSolvers.NonlinearSolvers
using GridapSolvers.LinearSolvers
using TimerOutputs

using Gridap.FESpaces
using GridapMultiSimO: jacobian, IterativeSolver, solve!
using WriteVTK
using Revise
 
function main()

  pname = "Static_Mechanical"
  meshfile = "cantilever.msh"
  simdir = datadir("sims", pname)
  setupfolder(simdir)

  geomodel = GmshDiscreteModel(datadir("models", meshfile))
  physmodel = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=0.0, ρ=1.0)

  # physmodel = LinearElasticity3D(λ=10.0, μ=1.0)

  # Setup integration
  order = 1
  degree = 2 * order + 1
  Ω = Triangulation(geomodel)
  dΩ = Measure(Ω, degree)

  # Dirichlet conditions 
  evolu(Λ) = 1.0
  dir_u_tags = ["fixedup"]
  dir_u_values = [[0.0, 0.0, 0.0]]
  dir_u_timesteps = [evolu]
  D_bc = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  # Neumann conditions 
  evolF(Λ) = Λ
  neu_F_tags = ["topcant"]
  neu_F_values = [[0.0, -0.001, 0.0]]
  neu_F_timesteps = [evolF]
  N_bc = NeumannBC(neu_F_tags, neu_F_values, neu_F_timesteps)
  dΓ = get_Neumann_dΓ(geomodel, N_bc, degree)

  #  FE spaces
  reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
  V = TestFESpace(geomodel, reffeu, D_bc, :H1)
  U = TrialFESpace(V, D_bc, 1.0)

  #  residual and jacobian function of load factor
  res(Λ) = (u, v) -> residual(physmodel, u, v, dΩ) + residual_Neumann(N_bc, v, dΓ, Λ)
  jac(Λ) = (u, du, v) -> jacobian(physmodel, u, du, v, dΩ)

  ls = LUSolver()
  nls_ = NewtonSolver(ls; maxiter=20, atol=1.e-12, rtol=1.e-8, verbose=true)

  comp_model = StaticNonlinearModel(res, jac, U, V, D_bc, dΩ; nls=nls_)

  post_model = PostMechanical(is_vtk=false, filepath=simdir)
  @timeit pname begin
    x = solve!(comp_model;  stepping=(nsteps=2, maxbisec=10), post=post_model)
  end
end

reset_timer!()
main()
print_timer()
 