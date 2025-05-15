using Gridap, GridapGmsh, HyperFEM, GridapSolvers, DrWatson, TimerOutputs
using GridapSolvers.NonlinearSolvers
using Gridap.FESpaces
using HyperFEM: jacobian, IterativeSolver, solve!
using WriteVTK
using Revise
using TimerOutputs
using Gridap.CellData

 
 
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
  dir_φ_values = [40000.0, 0.0]
  dir_φ_timesteps = [evolφ, evolφ]
  Dφ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

  evolu(Λ) = 1.0
  dir_u_tags_solid = ["usolid_fixed"]
  dir_u_values_solid = [[0.0, 0.0]]
  dir_u_timesteps_solid = [evolu]
  Du_solid = DirichletBC(dir_u_tags_solid, dir_u_values_solid, dir_u_timesteps_solid)

  dir_u_tags_air = ["usolid_fixed", "uair_fixed", "Interface"]
  dir_u_values_air = [[0.0, 0.0], [0.0, 0.0], x->x]
  dir_u_timesteps_air = [evolu, evolu,evolu]
  Du_air = DirichletBC(dir_u_tags_air, dir_u_values_air, dir_u_timesteps_air)
 


  dir_u_tags_air2 = ["Interface"]
  dir_u_values_air2 = [ (Λ)->(x->x*Λ)]
  dir_u_timesteps_air2 = [nothing]
  Du_air2 = DirichletBC(dir_u_tags_air2, dir_u_values_air2, dir_u_timesteps_air2)
 

  Vu_air2 = TestFESpace(Ωair, reffeu, Du_air2, conformity=:H1)
  Uu_air2 = TrialFESpace(Vu_air2, Du_air2, 0.5)
  Uu_air2.dirichlet_values


  map(f -> f(1.0), Du_air2.values)


  # FE spaces
  reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
  reffeφ = ReferenceFE(lagrangian, Float64, order)

  # Test FE Spaces
  Vφ = TestFESpace(geomodel, reffeφ, Dφ, conformity=:H1)
  Vu_solid = TestFESpace(Ωsolid, reffeu, Du_solid, conformity=:H1)
  Vu_air = TestFESpace(Ωair, reffeu, Du_air, conformity=:H1)


  # Trial FE Spaces and state variables
  Uφ = TrialFESpace(Vφ, Dφ, 1.0)
  φh⁺ = FEFunction(Uφ, zero_free_values(Uφ))
  Uφ⁻ = TrialFESpace(Vφ, Dφ, 1.0)
  φh⁻ = FEFunction(Uφ⁻, zero_free_values(Uφ⁻))

  Uu_solid = TrialFESpace(Vu_solid, Du_solid, 1.0)
  uh_solid⁺ = FEFunction(Uu_solid, zero_free_values(Uu_solid))
  Uu_solid⁻ = TrialFESpace(Vu_solid, Du_solid, 1.0)
  uh_solid⁻ = FEFunction(Uu_solid⁻, zero_free_values(Uu_solid⁻))

  Uu_air = TrialFESpace(Vu_air, Du_air, 1.0)
  uh_air⁺ = FEFunction(Uu_air, zero_free_values(Uu_air))
  Uu_air⁻ = TrialFESpace(Vu_air, Du_air, 1.0)
  uh_air⁻ = FEFunction(Uu_air⁻, zero_free_values(Uu_air⁻))



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
  C_uh_air(Λ) = uh_air⁻[1] + (uh_air⁺[1] - uh_air⁻[1]) * Λ
  C_φ(Λ) = φh⁻ + (φh⁺ - φh⁻) * Λ

  #   r=  assemble_vector((v)->∫((v.⁺⋅((DΨvacuum_mag[2] ∘ (F ∘ (∇(C_uh_air(1.0))'), ℋ₀ ∘ (∇(C_φ(1.0))))).⁻ ⋅ nΓsf.⁺)))dΓsf, Vu_solid )
  #   r2=  assemble_vector((v)->∫((v.⁺⋅((DΨvacuum_mag[2] ∘ (F ∘ (∇(C_uh_air(1.0))'), ℋ₀ ∘ (∇(C_φ(1.0))))).⁺ ⋅ nΓsf.⁺)))dΓsf, Vu_solid )

  uhsolid_int = interpolate_everywhere(uh_solid⁺, Usolid_int)
  uhair_int = interpolate_everywhere(Interpolable(uhsolid_int), Uair_int)
  uhsolid_int_(Λ) = Interpolable(interpolate_everywhere!(C_uh_solid(Λ), get_free_dof_values(uhsolid_int), uhsolid_int.dirichlet_values, Usolid_int))
  uhair_int_(Λ) = interpolate_everywhere!(uhsolid_int_(Λ), get_free_dof_values(uhair_int), uhair_int.dirichlet_values, Uair_int)


  Vu_air2 = TestFESpace(Ωair, reffeu, dirichlet_tags=["Interface", "uair_fixed"],conformity=:H1)
  Uu_air2 = TrialFESpace(Vu_air2,  VectorValue(0.0, 0.0))
  uh = FEFunction(Uair_int, zero_free_values(Uair_int))

 
Interface_coords_=reshape(Uu_air.dirichlet_values[Uu_air.space.dirichlet_dof_tag .==3],Int64(length(Uu_air.dirichlet_values[Uu_air.space.dirichlet_dof_tag .==3])/2),2)

mask = Uu_air.space.dirichlet_dof_tag .== 3
vals = Uu_air.dirichlet_values[mask]
Interface_coords_ = reshape(vals, :, 2)


Interface_coords = VectorValue.(eachrow(Interface_coords_))

v=evaluate(uhair_int_(1.0), Interface_coords)
 

length(Uu_air.space.fe_dof_basis.trian.model.grid.node_coordinates[1])
  


AA(1.0)

using Gridap.Arrays

AA= InterpolableBC(Uu_air, 3, uhair_int_ )


abstract type DirichletCoupling end

struct InterpolableBC{A,B,C}
coords::Vector{<:VectorValue}
Interpolable::Function
cache::C
function InterpolableBC(U::TrialFESpace, dirichlet_tags::Int64, Interpolable::Function)
  dim= length(U.space.fe_dof_basis.trian.model.grid.node_coordinates[1])
  mask = U.space.dirichlet_dof_tag .== dirichlet_tags
  vals = U.dirichlet_values[mask]
  Interface_coords_ = reshape(vals, :, dim)
  coords = VectorValue.(eachrow(Interface_coords_))
  v = evaluate(Interpolable(1.0), coords)
  bc_values = reduce(vcat,map(x->get_array(x),v))
  cache = (bc_values)
  new{typeof(coords),typeof(Interpolable),typeof(cache)}(coords, Interpolable, cache)
end

function (obj::InterpolableBC)(Λ::Float64=1.0)
  bc_values = obj.cache[1]
  bc_values .= reduce(vcat,map(x->get_array(x), evaluate(Interpolable(Λ), obj.coords)))
end

function get_bcvalues(obj::InterpolableBC)
  return obj.cache[1]
end

end
 
 