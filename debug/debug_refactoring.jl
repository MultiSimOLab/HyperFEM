using Mimosa
using Gridap
using DrWatson
using GridapGmsh
using GridapSolvers
using GridapSolvers.NonlinearSolvers
using CodeTracking

struct IntegrandWithMeasure{A,B<:Tuple}
    F  :: A
    dΩ :: B
  end
  (F::IntegrandWithMeasure)(args...) = F.F(args...,F.dΩ...)

abstract type ComputationalModel end
abstract type MultiphysicCoupling{Kind} end
abstract type Regime{Kind} end

  mesh_file = "cantilever.msh"
  PhysModel = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=0.0, ρ=1.0)
  Ψ, ∂Ψu, ∂Ψuu = PhysModel(DerivativeStrategy{:analytic}())
  model = GmshDiscreteModel(datadir("models", mesh_file))
  order = 1
  degree = 2 * order
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)

  evolu(Λ) = 1.0
  dir_u_tags = ["fixedup"]
  dir_u_values = [[0.0, 0.0, 0.0]]
  dir_u_timesteps = [evolu]
  dirichletbc = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  evolF(Λ) = Λ
  neu_F_tags = ["topcant"]
  neu_F_values = [[0.0, -0.1, 0.0]]
  neu_F_timesteps = [evolF]
  neumannbc = NeumannBC(neu_F_tags, neu_F_values, neu_F_timesteps)


  # NewtonRaphson parameters
  nr_show_trace = true
  nr_iter = 20
  nr_ftol = 1e-12

  # Incremental solver
  nsteps = 20
  nbisec = 10
 

  aa(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(v)))dΩ
 bb(x,y,z)=x*y*z
  TTT=(3,3)
  biform=IntegrandWithMeasure(bb,TTT)

ss(u,v)=biform(u,v)

@code_string ss(1,2)

  struct Mechanics{A,B,C,D,F} <: ComputationalModel
    res        :: A
    jac        :: B
    spaces     :: C
    dirichlet  :: D
    neumann    :: E
    istransient:: Bool
    cache      :: F
  
    function Mechanics(dΩ...;
      istransient= false,
      assem_U = SparseMatrixAssembler(U,V),
      nls::NonlinearSolver = NewtonSolver(LUSolver();maxiter=50,rtol=1.e-8,verbose=true),
    )
      res = IntegrandWithMeasure(res,dΩ)
      if isnothing(jac)
        jacf = (u,du,v,φh) -> jacobian(res,[u,v,φh],1)
      else
        jacf = (u,du,v,φh) -> jac(u,du,v,φh,dΩ...)
      end
      spaces = (U,V,V_φ,U_reg)
  
      ## Pullback cache
      uhd = zero(U)
      vecdata = collect_cell_vector(U_reg,∇(res,[uhd,uhd,φh],3))
      dudφ_vec = allocate_vector(assem_deriv,vecdata)
      plb_caches = (dudφ_vec,assem_deriv)
  
      ## Forward cache
      x = zero_free_values(U)
      _res(u,v) = res(u,v,φh)
      _jac(u,du,v) = jacf(u,du,v,φh)
      op = get_algebraic_operator(FEOperator(_res,_jac,U,V,assem_U))
      nls_cache = instantiate_caches(x,nls,op)
      fwd_caches = (nls,nls_cache,x,assem_U)
  
      ## Adjoint cache
      _jac_adj(du,v) = jacf(uhd,du,v,φh)
      adjoint_K  = assemble_adjoint_matrix(_jac_adj,assem_adjoint,U,V)
      adjoint_x  = allocate_in_domain(adjoint_K); fill!(adjoint_x,zero(eltype(adjoint_x)))
      adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
      adj_caches = (adjoint_ns,adjoint_K,adjoint_x,assem_adjoint)
      A, B, C = typeof(res), typeof(jacf), typeof(spaces)
      D, E, F = typeof(plb_caches), typeof(fwd_caches), typeof(adj_caches)
      return new{A,B,C,D,E,F}(res,jacf,spaces,plb_caches,fwd_caches,adj_caches)
    end
  end