using Gridap, GridapGmsh, HyperFEM, GridapSolvers, DrWatson, TimerOutputs
using GridapSolvers.NonlinearSolvers
using Gridap.FESpaces
using HyperFEM: jacobian, IterativeSolver, solve!
using WriteVTK
using Revise
using TimerOutputs
using Gridap.CellData


struct Anisotropy{A,B,C} <: Mechano
  modeliso::A
  modeltrans::B
  Kinematic::C
  function Anisotropy(; modeliso::Mechano, modeltrans::Mechano)
    Kinematic = modeliso.Kinematic
    new{typeof(modeliso),typeof(modeltrans),typeof(Kinematic)}(modeliso, modeltrans, Kinematic)
  end

  function (obj::Anisotropy)(Λ::Float64=1.0; kwargs...)
    Ψ1, ∂Ψu1, ∂Ψuu1 = obj.modeliso()
    Ψ2, ∂Ψu2, ∂Ψuu2 = obj.modeltrans()
    Ψ(F, N) = Ψ1(F) + Ψ2(F, N)
    ∂Ψu(F, N) = ∂Ψu1(F) + ∂Ψu2(F, N)
    ∂Ψuu(F, N) = ∂Ψuu1(F) + ∂Ψuu2(F, N)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end

end


function main()

  pname = "Staggered_MagnetoMech_vaccum_2D"
  meshfile = "rectangle_in_vacuum.msh"
  simdir = datadir("sims", pname)
  setupfolder(simdir)

  geomodel = GmshDiscreteModel(datadir("models", meshfile))

  μParams = [6456.9137547089595, 896.4633794151492,
    1.999999451256222,
    1.9999960497608036,
    11747.646562400318,
    0.7841068624959612, 1.5386288924587603]
  modelmech = NonlinearMoneyRivlin2D(λ=(μParams[1] + μParams[2]) * 1e2, μ1=μParams[1], μ2=μParams[2], α=μParams[3], β=μParams[4])

  # TransIso = TransverseIsotropy3D(μ=μParams[5], α=μParams[6], β=μParams[7])
  # modelmech = Anisotropy(modeliso=physmodel, modeltrans=TransIso)
  modelhard = HardMagnetic2D(μ=1.2566e-6, αr=40e-4 / 1.2566e-6, χe=0.0, χr=8.0; βmok=1.0, βcoup=1.0)
  modelmagneto_solid = MagnetoMechModel(Mechano=modelmech, Magneto=modelhard)
  model_vacuum_mech = IncompressibleNeoHookean2D(λ=μParams[1] * 1e1, μ=μParams[1], δ=0.1)
  model_vacuum_mag = MagnetoVacuumModel(Magneto=IdealMagnetic2D(μ=1.2566e-6, χe=0.0))

  # Domains
  order = 2
  degree = 2 * order
  bdegree = 4 * order
  Ωair = Interior(geomodel, tags=["Air"])
  dΩair = Measure(Ωair, degree)

  Ωsolid = Triangulation(geomodel, tags=["Solid"])
  dΩsolid = Measure(Ωsolid, degree)
  Ωpost = Triangulation(geomodel)

  Γair_int = BoundaryTriangulation(Ωair, tags="Interface")
  nair_int = get_normal_vector(Γair_int)
  dΓair_int = Measure(Γair_int, bdegree)

  Γsolid_int = BoundaryTriangulation(Ωsolid, tags="Interface")
  nsolid_int = get_normal_vector(Γsolid_int)
  dΓsolid_int = Measure(Γsolid_int, bdegree)

  Γsf = InterfaceTriangulation(Ωsolid, Ωair)
  nΓsf = get_normal_vector(Γsf)
  dΓsf = Measure(Γsf, bdegree)

  # Dirichlet conditions 

  # αload = 1.0 / 8.0
  # evolφ(Λ) = max(0.0, min(1.0, 1.0 / (1.0 - αload) * (Λ - αload)))
  evolφ(Λ) = Λ
  dir_φ_tags = ["φ_bottom", "φ_top"]
  dir_φ_values = [18000.0, 0.0]
  dir_φ_timesteps = [evolφ, evolφ]
  Dφ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

  evolu(Λ) = 1.0
  dir_u_tags_solid = ["usolid_fixed"]
  dir_u_values_solid = [[0.0, 0.0]]
  dir_u_timesteps_solid = [evolu]
  Du_solid = DirichletBC(dir_u_tags_solid, dir_u_values_solid, dir_u_timesteps_solid)

  dir_u_tags_air = ["usolid_fixed", "uair_fixed"]
  dir_u_values_air = [[0.0, 0.0], [0.0, 0.0]]
  dir_u_timesteps_air = [evolu, evolu]
  Du_air_ = DirichletBC(dir_u_tags_air, dir_u_values_air, dir_u_timesteps_air)
  Du_air = MultiFieldBC([Du_air_, NothingBC()])

  # FE spaces
  reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
  reffeφ = ReferenceFE(lagrangian, Float64, order)
  reffeλ = ReferenceFE(lagrangian, VectorValue{2,Float64}, order - 1)

  # Test FE Spaces
  Vφ = TestFESpace(geomodel, reffeφ, Dφ, conformity=:H1)
  Vu_solid = TestFESpace(Ωsolid, reffeu, Du_solid, conformity=:H1)
  Vu_air = TestFESpace(Ωair, reffeu, Du_air.BoundaryCondition[1], conformity=:H1)
  Vλ_air = TestFESpace(Γair_int, reffeλ, Du_air.BoundaryCondition[2], conformity=:H1)
  Vuλ_air = MultiFieldFESpace([Vu_air, Vλ_air])


  # Trial FE Spaces and state variables
  Uφ = TrialFESpace(Vφ, Dφ, 1.0)
  φh⁺ = FEFunction(Uφ, zero_free_values(Uφ))
  Uφ⁻ = TrialFESpace(Vφ, Dφ, 1.0)
  φh⁻ = FEFunction(Uφ⁻, zero_free_values(Uφ⁻))

  Uu_solid = TrialFESpace(Vu_solid, Du_solid, 1.0)
  uh_solid⁺ = FEFunction(Uu_solid, zero_free_values(Uu_solid))
  Uu_solid⁻ = TrialFESpace(Vu_solid, Du_solid, 1.0)
  uh_solid⁻ = FEFunction(Uu_solid⁻, zero_free_values(Uu_solid⁻))

  Uu_air = TrialFESpace(Vu_air, Du_air.BoundaryCondition[1], 1.0)
  Uλ_air = TrialFESpace(Vλ_air, Du_air.BoundaryCondition[2], 1.0)
  Uuλ_air = MultiFieldFESpace([Uu_air, Uλ_air])
  uλ_air⁺ = FEFunction(Uuλ_air, zero_free_values(Uuλ_air))

  Uu_air⁻ = TrialFESpace(Vu_air, Du_air.BoundaryCondition[1], 1.0)
  Uλ_air⁻ = TrialFESpace(Vλ_air, Du_air.BoundaryCondition[2], 1.0)
  Uuλ_air⁻ = MultiFieldFESpace([Uu_air⁻, Uλ_air⁻])
  uλ_air⁻ = FEFunction(Uuλ_air⁻, zero_free_values(Uuλ_air⁻))



  # Interface Coupling
  Uair_int = FESpace(Γair_int, reffeu)
  Usolid_int = FESpace(Γsolid_int, reffeu)

  # DΨsolid(Λ) = modelmagneto_solid(min(1.0, (1.0 / αload) * Λ))
  DΨsolid(Λ) = modelmagneto_solid(1.0)
  DΨvacuum_mag = model_vacuum_mag(1.0)
  DΨvacuum_mech = model_vacuum_mech(1.0)

  F, _, _ = get_Kinematics(modelmagneto_solid.Mechano.Kinematic; Λ=1.0)
  ℋ₀ = get_Kinematics(modelmagneto_solid.Magneto.Kinematic; Λ=1.0)

  C_uh_solid(Λ) = uh_solid⁻ + (uh_solid⁺ - uh_solid⁻) * Λ
  C_uh_air(Λ) = uλ_air⁻[1] + (uλ_air⁺[1] - uλ_air⁻[1]) * Λ
  C_φ(Λ) = φh⁻ + (φh⁺ - φh⁻) * Λ

  #   r=  assemble_vector((v)->∫((v.⁺⋅((DΨvacuum_mag[2] ∘ (F ∘ (∇(C_uh_air(1.0))'), ℋ₀ ∘ (∇(C_φ(1.0))))).⁻ ⋅ nΓsf.⁺)))dΓsf, Vu_solid )
  #   r2=  assemble_vector((v)->∫((v.⁺⋅((DΨvacuum_mag[2] ∘ (F ∘ (∇(C_uh_air(1.0))'), ℋ₀ ∘ (∇(C_φ(1.0))))).⁺ ⋅ nΓsf.⁺)))dΓsf, Vu_solid )

  uhsolid_int = interpolate_everywhere(uh_solid⁺, Usolid_int)
  uhair_int = interpolate_everywhere(Interpolable(uhsolid_int), Uair_int)
  uhsolid_int_(Λ) = Interpolable(interpolate_everywhere!(C_uh_solid(Λ), get_free_dof_values(uhsolid_int), uhsolid_int.dirichlet_values, Usolid_int))
  uhair_int_(Λ) = interpolate_everywhere!(uhsolid_int_(Λ), get_free_dof_values(uhair_int), uhair_int.dirichlet_values, Uair_int)

  Vn = TestFESpace(geomodel, reffeu)
  Nh = interpolate_everywhere(VectorValue(1.0, 0.0), Vn)

  # Magneto in vacuum
  res_mag(Λ) = (φ, vφ) -> -1.0 * ∫((∇(vφ) ⋅ (DΨsolid(Λ)[3] ∘ (F ∘ (∇(C_uh_solid(Λ))'), ℋ₀ ∘ (∇(φ)), Nh))))dΩsolid -
                          ∫((∇(vφ) ⋅ (DΨvacuum_mag[3] ∘ (F ∘ (∇(C_uh_air(Λ))'), ℋ₀ ∘ (∇(φ))))))dΩair

  jac_mag(Λ) = (φ, dφ, vφ) -> ∫(∇(vφ)' ⋅ ((DΨsolid(Λ)[6] ∘ (F ∘ (∇(C_uh_solid(Λ))'), ℋ₀ ∘ (∇(φ)), Nh)) ⋅ ∇(dφ)))dΩsolid +
                              ∫(∇(vφ)' ⋅ ((DΨvacuum_mag[6] ∘ (F ∘ (∇(C_uh_air(Λ))'), ℋ₀ ∘ (∇(φ)))) ⋅ ∇(dφ)))dΩair

  # Solid
  res_mech(Λ) = (u, v) -> ∫((∇(v)' ⊙ (DΨsolid(Λ)[2] ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(C_φ(Λ))), Nh))))dΩsolid - ∫((v.⁺ ⋅ ((DΨvacuum_mag[2] ∘ (F ∘ (∇( C_uh_solid(Λ))'), ℋ₀ ∘ (∇(C_φ(Λ))))).⁻ ⋅ nΓsf.⁺)))dΓsf
  jac_mech(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((DΨsolid(Λ)[4] ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(C_φ(Λ))), Nh)) ⊙ (∇(du)')))dΩsolid

  # Vacuum mechanics
  κr = 1e-3 * 0.1^2 * model_vacuum_mech.λ
  res_vacmech(Λ) = ((u, λ), (v, vλ)) -> ∫((∇(v)' ⊙ (DΨvacuum_mech[2] ∘ (F ∘ (∇(u)')))))dΩair + ∫(κr * ∇∇(u) ⊙ ∇∇(v))dΩair +
                                        ∫(λ ⋅ v)dΓair_int + ∫(vλ ⋅ (u - uhair_int_(Λ)))dΓair_int

  jac_vacmech(Λ) = ((u, λ), (du, dλ), (v, vλ)) -> ∫(∇(v)' ⊙ ((DΨvacuum_mech[3] ∘ (F ∘ (∇(u)'))) ⊙ (∇(du)')))dΩair +
                                                  ∫(κr * ∇∇(du) ⊙ ∇∇(v))dΩair +
                                                  ∫(dλ ⋅ v)dΓair_int + ∫(vλ ⋅ du)dΓair_int
  # nonlinear solver Vacuum Magneto
  ls = LUSolver()
  nls_ = NewtonSolver(ls; maxiter=20, atol=1.e-9, rtol=1.e-8, verbose=true)
  nls_vacmech = NewtonSolver(ls; maxiter=20, atol=1.e-7, rtol=1.e-7, verbose=true)

  comp_model_mag = StaticNonlinearModel(res_mag, jac_mag, Uφ, Vφ, Dφ; nls=nls_, xh=φh⁺)
  comp_model_mech = StaticNonlinearModel(res_mech, jac_mech, Uu_solid, Vu_solid, Du_solid; nls=nls_, xh=uh_solid⁺)
  comp_model_vacmech = StaticNonlinearModel(res_vacmech, jac_vacmech, Uuλ_air, Vuλ_air, Du_air; nls=nls_vacmech, xh=uλ_air⁺)
  comp_model = StaggeredModel((comp_model_mag, comp_model_mech, comp_model_vacmech), (φh⁺, uh_solid⁺, uλ_air⁺), (φh⁻, uh_solid⁻, uλ_air⁻))

  function driverpost_mech(post; Ω=Ωsolid, U=Uu_solid)
    # get from postprocessor 
    state = post.comp_model.caches[3]
    Λ_ = post.iter
    Λ = post.Λ[Λ_]

    xh = FEFunction(U, state)
    pvd = post.cachevtk[3]
    filePath = post.cachevtk[2]

    if post.cachevtk[1]
      Λstring = replace(string(round(Λ, digits=2)), "." => "_")
      pvd[Λ_] = createvtk(Ω,
        filePath * "/_Λmech_" * Λstring * "_TIME_$Λ_" * ".vtu",
        cellfields=["u" => xh])

    end
  end

  function driverpost_mag(post; Ω=Ωpost, U=Uφ)
    # get from postprocessor 
    state = post.comp_model.caches[3]
    Λ_ = post.iter
    Λ = post.Λ[Λ_]

    xh = FEFunction(U, state)

    pvd = post.cachevtk[3]
    filePath = post.cachevtk[2]

    if post.cachevtk[1]
      Λstring = replace(string(round(Λ, digits=2)), "." => "_")
      pvd[Λ_] = createvtk(Ω,
        filePath * "/_Λmag_" * Λstring * "_TIME_$Λ_" * ".vtu",
        cellfields=["φ" => xh])

    end
  end

  function driverpost_vacmech(post; Ω=Ωair, U=Uuλ_air, φ=φh⁺)
    # get from postprocessor 
    state = post.comp_model.caches[3]
    Λ_ = post.iter
    Λ = post.Λ[Λ_]

    xh = FEFunction(U, state)

    pvd = post.cachevtk[3]
    filePath = post.cachevtk[2]

    if post.cachevtk[1]
      Λstring = replace(string(round(Λ, digits=2)), "." => "_")
      pvd[Λ_] = createvtk(Ω,
        filePath * "/_Λmagmech_" * Λstring * "_TIME_$Λ_" * ".vtu",
        cellfields=["u" => xh[1], "-∇(φh)" => -∇(φ)])

    end
  end

  post_model_mech = PostProcessor(comp_model_mech, driverpost_mech; is_vtk=true, filepath=simdir)
  post_model_mag = PostProcessor(comp_model_mag, driverpost_mag; is_vtk=true, filepath=simdir)
  post_model_vacmech = PostProcessor(comp_model_vacmech, driverpost_vacmech; is_vtk=true, filepath=simdir)

  args_mag = Dict(:stepping => (nsteps=1, maxbisec=5), :post => post_model_mag)
  args_mech = Dict(:stepping => (nsteps=20, maxbisec=5), :post => post_model_mech )
  args_vacmech = Dict(:stepping => (nsteps=2, maxbisec=5), :post => post_model_vacmech)
  args = (args_mag, args_mech, args_vacmech)



#   x⁺, x⁻ = comp_model.caches
#   map((x) -> TrialFESpace!(x.spaces[1], x.dirichlet, 0.0), comp_model.compmodels)
#   map((x, y) -> TrialFESpace!(x.fe_space, y.dirichlet, 0.0), comp_model.state⁻, comp_model.compmodels)


#   nsteps = 1000
#   flagconv = 1 # convergence flag 0 (max bisections) 1 (max steps)
#   ∆Λ = 1.0 / nsteps
#   nbisect = 0
#   time = 0
#   while time < nsteps
# @show time
#     stevol(Λ) = ∆Λ * (Λ + time)
#     updateBC!(comp_model.compmodels[1].dirichlet, comp_model.compmodels[1].dirichlet.caches, [stevol for _ in 1:length(comp_model.compmodels[1].dirichlet.caches)])
#     updateBC!(comp_model.compmodels[2].dirichlet, comp_model.compmodels[2].dirichlet.caches, [stevol for _ in 1:length(comp_model.compmodels[2].dirichlet.caches)])
#     updateBC!(comp_model.compmodels[3].dirichlet.BoundaryCondition[1], comp_model.compmodels[3].dirichlet.BoundaryCondition[1].caches, [stevol for _ in 1:length(comp_model.compmodels[3].dirichlet.BoundaryCondition[1].caches)])
 
#     # map(x -> updateBC!(x.dirichlet, stevol), comp_model.compmodels)
#     map((x) -> TrialFESpace!(x.spaces[1], x.dirichlet, 1.0), comp_model.compmodels)
#     # solve magneto
#     solve!(comp_model_mag; args[1]...)
#       r=  assemble_vector((v)->∫((v.⁺⋅((DΨvacuum_mag[2] ∘ (F ∘ (∇(C_uh_air(1.0))'), ℋ₀ ∘ (∇(C_φ(1.0))))).⁻ ⋅ nΓsf.⁺)))dΓsf, Vu_solid )
# @show norm(r)
#     # solve solid
#     solve!(comp_model_mech; args[2]...)
 
#     #solve vacuum
#     solve!(comp_model_vacmech; args[3]...)
 
#     map((x, y) -> TrialFESpace!(x.fe_space, y.dirichlet, 1.0), comp_model.state⁻, comp_model.compmodels)
#     map((x, y) -> x .= y, x⁻, x⁺)
#     time = time + 1
 
#   end

  @timeit pname begin
    solve!(comp_model; stepping=(nsteps=3, maxbisec=15), kargsolve=args)
  end

  # writevtk(Ωair, simdir * "/_post_.vtu",
  #   cellfields=["Uu_air" => uλ_air⁺[1]])
end


reset_timer!()
main()
print_timer()
 