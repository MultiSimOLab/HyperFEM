using HyperFEM
using HyperFEM: jacobian, solve!
using HyperFEM.ComputationalModels.PostMetrics
using HyperFEM.ComputationalModels.CartesianTags
using Gridap, GridapGmsh, GridapSolvers, DrWatson
using GridapSolvers.NonlinearSolvers
using Gridap.FESpaces

folder = joinpath(@__DIR__, "results", "ThermoViscoElectric")
setupfolder(folder; remove=".vtu")


long  = 100  # m
width = 100  # m
thick = 1 # m
domain = (0.0, long, 0.0, width, 0.0, thick)
partition = (8, 2, 2)
geometry = CartesianDiscreteModel(domain, partition)
labels = get_face_labeling(geometry)

add_tag_from_tags!(labels, "fixed", [CartesianTags.edgeX00 ;CartesianTags.edgeX10 ;CartesianTags.edge0Y0 ;CartesianTags.edge1Y0;CartesianTags.corner000; CartesianTags.corner100 ; CartesianTags.corner010 ; CartesianTags.corner110 ])
add_tag_from_tags!(labels, "bottom", CartesianTags.faceZ0)
add_tag_from_tags!(labels, "top", CartesianTags.faceZ1)


# Constitutive model
hyper_elastic_model = NeoHookean3D(λ=10000.0, μ=2.0e5)
viscous_branch_1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0.0, μ=4.0e5); τ=2.0)
viscous_branch_2 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0.0, μ=2.0e4); τ=10.0)
visco_elastic_model = GeneralizedMaxwell(hyper_elastic_model, viscous_branch_1, viscous_branch_2)
elec_model = IdealDielectric(ε=3.5416e-11)
therm_model = ThermalModel(Cv=100.0, θr=293.15, α=1.1631, κ=10.0)
cons_model = ThermoElectroMech_Bonet(therm_model, elec_model, visco_elastic_model)
ku = Kinematics(Mechano, Solid)
ke = Kinematics(Electro, Solid)
kt = Kinematics(Thermo, Solid)
F, _... = get_Kinematics(ku)
E       = get_Kinematics(ke)

# Setup integration
order = 1
degree = 2 * order
Ω = Triangulation(geometry)
dΩ = Measure(Ω, degree)
t_end = 1.0  # s
Δt = 0.005   # s

# Dirichlet boundary conditions 
evolu(Λ) = 1.0
dir_u_tags = ["fixed"]
dir_u_values = [[0.0, 0.0, 0.0]]
dir_u_timesteps = [evolu]
dirichlet_u = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

evolφ(Λ) = Λ
dir_φ_tags = ["bottom", "top"]
dir_φ_values = [0.0, 0.002]
dir_φ_timesteps = [evolφ, evolφ]
dirichlet_φ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

dirichlet_θ = NothingBC()
dirichlet_bc = MultiFieldBC([dirichlet_u, dirichlet_φ, dirichlet_θ])

# Finite Elements
reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
reffeφ = ReferenceFE(lagrangian, Float64, order)
reffeθ = ReferenceFE(lagrangian, Float64, order)

# Test FE Spaces
Vu = TestFESpace(geometry, reffeu, dirichlet_u, conformity=:H1)
Vφ = TestFESpace(geometry, reffeφ, dirichlet_φ, conformity=:H1)
Vθ = TestFESpace(geometry, reffeθ, dirichlet_θ, conformity=:H1)

# Trial FE Spaces and state variables
Uu  = TrialFESpace(Vu, dirichlet_u)
Uφ  = TrialFESpace(Vφ, dirichlet_φ)
Uθ  = TrialFESpace(Vθ, dirichlet_θ)
uh⁺ = FEFunction(Uu, zero_free_values(Uu))
φh⁺ = FEFunction(Uφ, zero_free_values(Uφ))
θh⁺ = FEFunction(Uθ, zero_free_values(Uθ))

Uu⁻ = TrialFESpace(Vu, dirichlet_u)
Uφ⁻ = TrialFESpace(Vφ, dirichlet_φ)
Uθ⁻ = TrialFESpace(Vθ, dirichlet_θ)
uh⁻ = FEFunction(Uu⁻, zero_free_values(Uu⁻))
φh⁻ = FEFunction(Uφ⁻, zero_free_values(Uφ⁻))
θh⁻ = FEFunction(Uθ⁻, zero_free_values(Uθ⁻))

Uη⁻ = TrialFESpace(Vθ)
ηh⁻ = FEFunction(Uη⁻, zero_free_values(Uη⁻))

UD⁻ = TrialFESpace(Vθ)
Dh⁻ = FEFunction(UD⁻, zero_free_values(UD⁻))

Fh = F∘∇(uh⁺)'
Fh⁻ = F∘∇(uh⁻)'
A = initializeStateVariables(cons_model, dΩ)

# =================================
# Weak forms: residual and jacobian
# =================================

Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂EE, ∂∂Ψ∂θθ, ∂∂Ψ∂FE, ∂∂Ψ∂Fθ, ∂∂Ψ∂Eθ, _ = cons_model(Δt=Δt)
D, ∂D∂θ = Dissipation(cons_model, Δt)
η(x...) = -∂Ψ∂θ(x...)
∂η∂θ(x...) = -∂∂Ψ∂θθ(x...)
κ = cons_model.thermo.κ

Mechano_coupling(Λ) = uh⁻ + (uh⁺ - uh⁻) * Λ
Electro_coupling(Λ) = φh⁻ + (φh⁺ - φh⁻) * Λ
Thermo_coupling(Λ)  = θh⁻ + (θh⁺ - θh⁻) * Λ

# Electro
res_elec(Λ) = (φ, vφ) -> residual(cons_model, Electro, (ku, ke, kt), (Mechano_coupling(Λ), φ, Thermo_coupling(Λ)), vφ, dΩ, 0.0, Fh⁻, A...; Δt=Δt)
jac_elec(Λ) = (φ, dφ, vφ) -> jacobian(cons_model, Electro, (ku, ke, kt), (Mechano_coupling(Λ), φ, Thermo_coupling(Λ)), dφ, vφ, dΩ, 0.0, Fh⁻, A...; Δt=Δt)

# Mechano
res_mec(Λ) = (u, v) -> residual(cons_model, Mechano, (ku, ke, kt), (u, Electro_coupling(Λ), Thermo_coupling(Λ)), v, dΩ, 0.0, Fh⁻, A...; Δt=Δt)
jac_mec(Λ) = (u, du, v) -> jacobian(cons_model, Mechano, (ku, ke, kt), (u, Electro_coupling(Λ), Thermo_coupling(Λ)), du, v, dΩ, 0.0, Fh⁻, A...; Δt=Δt)

# Thermo
res_therm(Λ) = (θ, vθ) -> begin
  uhᵞ = Mechano_coupling(Λ)
  φhᵞ = Electro_coupling(Λ)
  ∫( 1/Δt*(θ*η∘(F∘∇(uhᵞ), E∘∇(φhᵞ), θ, Fh⁻, A...) - θh⁻*ηh⁻)*vθ +
    -1/Δt*0.5*(η∘(F∘∇(uhᵞ), E∘∇(φhᵞ), θ, Fh⁻, A...) + ηh⁻)*(θ - θh⁻)*vθ +
    -0.5*(D∘(F∘∇(uhᵞ), E∘∇(φhᵞ), θ, Fh⁻, A...) + Dh⁻)*vθ +
     0.5*(κ * ∇(θ)·∇(vθ)) + 0.5*(κ * ∇(θh⁻)·∇(vθ))
  )dΩ
end
jac_therm(Λ) = (θ, dθ, vθ) -> begin
  uhᵞ = Mechano_coupling(Λ)
  φhᵞ = Electro_coupling(Λ)
  ∫( 1/Δt*(η∘(F∘∇(uhᵞ), E∘∇(φhᵞ), θ, Fh⁻, A...) + θ*∂η∂θ∘(F∘∇(uhᵞ), E∘∇(φhᵞ), θ, Fh⁻, A...))*dθ*vθ +
    -1/Δt*0.5*(∂η∂θ∘(F∘∇(uhᵞ), E∘∇(φhᵞ), θ, Fh⁻, A...)*(θ - θh⁻) + (η∘(F∘∇(uhᵞ), E∘∇(φhᵞ), θ, Fh⁻, A...) + ηh⁻))*dθ*vθ +
    -0.5*∂D∂θ∘(F∘∇(uhᵞ), E∘∇(φhᵞ), θ, Fh⁻, A...)*dθ*vθ +
     0.5*κ*∇(dθ)·∇(vθ)
  )dΩ
end

# nonlinear solver electro
ls = LUSolver()
nls = NewtonSolver(ls; maxiter=20, atol=1e-6, rtol=1e-6, verbose=true)
comp_model_elec = StaticNonlinearModel(res_elec, jac_elec, Uφ, Vφ, dirichlet_φ; nls=nls, xh=φh⁺, xh⁻=φh⁻)

# nonlinear solver mechano
comp_model_mec = StaticNonlinearModel(res_mec, jac_mec, Uu, Vu, dirichlet_u; nls=nls, xh=uh⁺, xh⁻=uh⁻)

# nonlinear solver thermo
comp_model_mec = StaticNonlinearModel(res_therm, jac_therm, Uθ, Vθ, dirichlet_θ; nls=nls, xh=θh⁺, xh⁻=θh⁻)

# nonlinear staggered model
comp_model = StaggeredModel((comp_model_elec, comp_model_mec), (φh⁺, uh⁺, θh⁺), (φh⁻, uh⁻, θh⁻))

count = Ref{Int}(0)

# Postprocessor to save results
function driverpost(post)
  step = post.iter
  pvd = post.cachevtk[3]
  filePath = post.cachevtk[2]
  count[] += 1

  push!(uz, component_LInf(uh⁺, :z, Ω))
  if post.cachevtk[1]
      pvd[step] = createvtk(Ω, filePath * "/TIME_$(count[])" * ".vtu", cellfields=["u" => uh⁺, "φ" => φh⁺])
  end
  updateStateVariables!(A, cons_model, Δt, F∘∇(uh⁺), F∘∇(uh⁻))
end

post_model = PostProcessor(comp_model_mec, driverpost; is_vtk=true, filepath=folder)

args_elec = Dict(:stepping => (nsteps=1, maxbisec=1))
args_mec = Dict(:stepping => (nsteps=1, maxbisec=1))
args_therm = Dict(:stepping => (nsteps=1, maxbisec=1), :post=>post_model)
args = (args_elec, args_mec, args_therm)

t =  [0:Δt:t_end-Δt...]
uz = Float64[]
nsteps = t_end / Δt
solve!(comp_model; stepping=(nsteps=nsteps, nsubsteps=1, maxbisec=1), kargsolve=args)

plot(t, uz)

@assert uz[end] ≈ 1.39059921e-5
@assert norm(uz) ≈ 8.79961900e-5
