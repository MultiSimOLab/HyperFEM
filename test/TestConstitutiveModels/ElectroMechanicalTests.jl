using Gridap.TensorValues
using HyperFEM.PhysicalModels


const ∇φ = VectorValue(1.0:3.0...)
const ∇u = TensorValue(1.0:9.0...) * 1e-3
const ∇un = TensorValue(1.0:9.0...) * 5e-4


@testset "ElectroMechano" begin
  modelMR = MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0)
  modelID = IdealDielectric(ε=4.0)
  modelelectro = ElectroMechModel(mechano=modelMR, electro=modelID)
  Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = modelelectro()
  F, _, _ = get_Kinematics(modelMR.Kinematic)
  E = get_Kinematics(modelID.Kinematic)

  @test Ψ(F(∇u), E(∇φ)) == -27.514219755428428
  @test norm(∂Ψu(F(∇u), E(∇φ))) == 47.42294370458073
  @test norm(∂Ψφ(F(∇u), E(∇φ))) == 14.707913034885005
  @test norm(∂Ψuu(F(∇u), E(∇φ))) == 131.10069227603947
  @test norm(∂Ψφu(F(∇u), E(∇φ))) == 39.03656526472973
  @test norm(∂Ψφφ(F(∇u), E(∇φ))) == 6.964428025226914
end


@testset "FlexoElectroMechanics" begin

  # Constitutive models
  ∇umacro = TensorValue(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) * 1e-2
  ∇u1 = 1e-1 * TensorValue(1, 2, 3, 4, 5, 6, 7, 8, 9)
  Emacro = VectorValue(0.0, 0.0, sqrt((1.0 + 5.0) / (1.0 + 5.0)) * 0.1)
  A = TensorValue{3,9,Float64,27}(0.0013981268088158305, 0.0008195783555664171,
    0.0016562357569609649, 0.0008406006468943406, 0.0009224862278332126, 0.001155322042969417,
    0.0005129360612093835, 0.0012909164959851265, 0.001152698427032676, 0.0008406006468943406,
    0.0009224862278332126, 0.001155322042969417, 0.00034502469077903774, 0.00021859521770246592,
    0.0017683239822952042, 0.0009471782270005929, 0.001800950730156155, 0.0009587801251013468,
    0.0005129360612093835, 0.0012909164959851265, 0.001152698427032676, 0.0009471782270005929,
    0.001800950730156155, 0.0009587801251013468, 0.0008421896546088605, 0.0007114140805416631,
    0.001245006227831607)
  Kin_mec = EvolutiveKinematics(Mechano; F=(t) -> ((∇u1, x) -> ∇u1 + one(∇u1) + t * ∇umacro + t * (A ⊙ x)))
  Kin_elec = EvolutiveKinematics(Electro; E=(t) -> ((∇φ) -> -∇φ + t * Emacro))

  physmec = MooneyRivlin3D(λ=10.0, μ1=1.0, μ2=1.0, Kinematic=Kin_mec)
  physelec = IdealDielectric(ε=1.0, Kinematic=Kin_elec)
  physmodel = FlexoElectroModel(mechano=physmec, electro=physelec, κ=1000.0)

  Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ, Φ = physmodel(1.0)

  F, _, _ = get_Kinematics(physmec.Kinematic; Λ=1.0)
  E = get_Kinematics(physelec.Kinematic; Λ=1.0)
  X = VectorValue(2.4, 1.9, 3.3)

  @test (Ψ(F(∇u1, X), E(∇φ))) == 13.408299698687056
  @test norm(∂Ψu(F(∇u1, X), E(∇φ))) == 58.375248703633474
  @test norm(∂Ψφ(F(∇u1, X), E(∇φ))) == 1.2365693126167825
  @test norm(∂Ψuu(F(∇u1, X), E(∇φ))) == 208.40589433833898
  @test norm(∂Ψφφ(F(∇u1, X), E(∇φ))) == 3.8963298254031042
  @test norm(∂Ψφu(F(∇u1, X), E(∇φ))) == 5.910650247536949
end


@testset "ViscoElectricModel" begin
  hyper_elastic = NeoHookean3D(λ=1000., μ=10.)
  short_term = IncompressibleNeoHookean3D(μ=5., λ=0.)
  viscous_branch1 = ViscousIncompressible(short_term, 6.)
  visco_elastic = GeneralizedMaxwell(hyper_elastic, viscous_branch1)
  dielectric = IdealDielectric(ε=1.0)
  model = ElectroMechModel(visco_elastic, dielectric)
  F, _, _ = get_Kinematics(model.mechano.Kinematic)
  E       = get_Kinematics(model.electro.Kinematic)
  Uvn = TensorValue(1.,2.,3.,2.,4.,5.,3.,5.,6.) * 2e-4 + I3
  Uvn *= det(Uvn)^(-1/3)
  λvn = 1e-3
  Avn = VectorValue(Uvn..., λvn)
  Ψ, ∂Ψu, ∂Ψuu = model(Δt=0.01)
  @show ∂Ψu(F(∇u), F(∇un), E(∇φ), Avn)
  @show ∂Ψuu(F(∇u), F(∇un), E(∇φ), Avn)
end


@testset "ViscoElectricModel2Branch" begin
  hyper_elastic = NeoHookean3D(λ=1000., μ=10.)
  short_term = IncompressibleNeoHookean3D(μ=5., λ=0.)
  viscous_branch1 = ViscousIncompressible(short_term, 6.)
  viscous_branch2 = ViscousIncompressible(short_term, 60.)
  visco_elastic = GeneralizedMaxwell(hyper_elastic, viscous_branch1, viscous_branch2)
  dielectric = IdealDielectric(ε=1.0)
  model = ElectroMechModel(visco_elastic, dielectric)
  F, _, _ = get_Kinematics(model.mechano.Kinematic)
  E       = get_Kinematics(model.electro.Kinematic)
  Uvn = TensorValue(1.,2.,3.,2.,4.,5.,3.,5.,6.) * 2e-4 + I3
  Uvn *= det(Uvn)^(-1/3)
  λvn = 1e-3
  Avn = VectorValue(Uvn..., λvn)
  Ψ, ∂Ψu, ∂Ψuu = model(Δt=0.01)
  @show ∂Ψu(F(∇u), F(∇un), E(∇φ), Avn, Avn)
  @show ∂Ψuu(F(∇u), F(∇un), E(∇φ), Avn, Avn)
end
