using Gridap, GridapGmsh, HyperFEM, GridapSolvers, DrWatson, TimerOutputs
using GridapSolvers.NonlinearSolvers
using Gridap.FESpaces
using HyperFEM: jacobian, IterativeSolver, solve!
using WriteVTK
using Revise
using TimerOutputs
using HyperFEM


pname = "Static_ElectroMechanical_staggered"
meshfile = "ex2_mesh.msh"
simdir = datadir("sims", pname)
setupfolder(simdir)

geomodel = GmshDiscreteModel(datadir("models", meshfile))

physmodel_mec = NeoHookean3D(λ=10.0, μ=1.0)
physmodel_elec = IdealDielectric(ε=1.0)
physmodel = ElectroMechModel(Mechano=physmodel_mec, Electro=physmodel_elec)

# Setup integration
order = 2
degree = 2 * order
Ω = Triangulation(geomodel)
dΩ = Measure(Ω, degree)

# Dirichlet conditions 
evolu(Λ) = 1.0
dir_u_tags = ["fixedup"]
dir_u_values = [[0.0, 0.0, 0.0]]
dir_u_timesteps = [evolu]
Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

electrodes = (x)->(x[1] > 4 ? 0.15 : 0.05)
evolφ(Λ) = Λ
dir_φ_tags = ["midsuf", "topsuf"]
dir_φ_values = [0.0, electrodes]
dir_φ_timesteps = [evolφ, evolφ]
Dφ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

# FE spaces
reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
reffeφ = ReferenceFE(lagrangian, Float64, order)

# Test FE Spaces
Vu = TestFESpace(geomodel, reffeu, Du, :H1)
Vφ = TestFESpace(geomodel, reffeφ, Dφ, :H1)

# Trial FE Spaces and state variables
Uu = TrialFESpace(Vu, Du, 1.0)
uh⁺ = FEFunction(Uu, zero_free_values(Uu))

Uu⁻ = TrialFESpace(Vu, Du, 1.0)
uh⁻ = FEFunction(Uu⁻, zero_free_values(Uu⁻))

Uφ = TrialFESpace(Vφ, Dφ, 1.0)
φh⁺ = FEFunction(Uφ, zero_free_values(Uφ))

Uφ⁻ = TrialFESpace(Vφ, Dφ, 1.0)
φh⁻ = FEFunction(Uφ⁻, zero_free_values(Uφ⁻))
    
# Electro
Mechano_coupling(Λ) = uh⁻ + (uh⁺ - uh⁻) * Λ
res_elec(Λ) = (φ, vφ) -> residual(physmodel, Electro, (Mechano_coupling(Λ), φ), vφ, dΩ)
jac_elec(Λ) = (φ, dφ, vφ) -> jacobian(physmodel, Electro, (Mechano_coupling(Λ), φ), dφ, vφ, dΩ)

# Mechano
Electro_coupling(Λ) = φh⁻ + (φh⁺ - φh⁻) * Λ
res_mec(Λ) = (u, v) -> residual(physmodel, Mechano, (u, Electro_coupling(Λ)), v, dΩ)
jac_mec(Λ) = (u, du, v) -> jacobian(physmodel, Mechano, (u, Electro_coupling(Λ)), du, v, dΩ)

# nonlinear solver electro
ls = LUSolver()
nls_ = NewtonSolver(ls; maxiter=20, atol=1.e-10, rtol=1.e-8, verbose=true)
comp_model_elec = StaticNonlinearModel(res_elec, jac_elec, Uφ, Vφ, Dφ, dΩ; nls=nls_, xh=φh⁺)

# nonlinear solver mechano
comp_model_mec = StaticNonlinearModel(res_mec, jac_mec, Uu, Vu, Du, dΩ; nls=nls_, xh=uh⁺)

# nonlinear staggered model
comp_model= StaggeredModel((comp_model_elec,comp_model_mec), (φh⁺,uh⁺), (φh⁻,uh⁻))

args_elec = Dict(:stepping => (nsteps=1,maxbisec=5))
args_mec  = Dict(:stepping => (nsteps=5,maxbisec=5))
args=(args_elec,args_mec)

solve!(comp_model; stepping=(nsteps=5, maxbisec=15), kargsolve=args)

writevtk(Ω, simdir * "/result2_end", cellfields=["φh" => φh⁺, "uh" => uh⁺])

 