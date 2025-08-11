using Gridap
using HyperFEM: jacobian


@testset "Assembly Jacobian ThermoMechanics" begin
    modelMR = MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0)
    modelT = ThermalModel(Cv=1.0, θr=1.0, α=2.0, κ=1.0)
    f(δθ::Float64)::Float64 = (δθ + 1.0) / 1.0
    df(δθ::Float64)::Float64 = 1.0
    modelTM = ThermoMechModel(Thermo=modelT, Mechano=modelMR, fθ=f, dfdθ=df)

    partition = (1, 1, 1)
    pmin = Point(0.0, 0.0, 0.0)
    L = VectorValue(1, 1, 1)
    pmax = pmin + L
    model = CartesianDiscreteModel(pmin, pmax, partition)
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "support0", [1, 2, 3, 4, 5, 6, 7, 8])

    order = 1
    reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
    reffeθ = ReferenceFE(lagrangian, Float64, order)

    #Define test FESpaces
    Vu = TestFESpace(model, reffeu, labels=labels, conformity=:H1)
    Vθ = TestFESpace(model, reffeθ, labels=labels, conformity=:H1)
    V = MultiFieldFESpace([Vu, Vθ])
    U = V
    #Setup integration
    degree = 2 * order
    Ωₕ = Triangulation(model)
    dΩ = Measure(Ωₕ, 2)

    uh = FEFunction(Vu, zeros(Float64, num_free_dofs(Vu)))
    θ(x) = x[1] - 1.0
    θh = interpolate_everywhere(θ, Vθ)

    function jach(uh, θh)
        jac((du, dθ), (v, vθ)) = jacobian(modelTM, (uh, θh), (du, dθ), (v, vθ), dΩ)
    end
    function jac_mech(uh, θh)
        jac(du, v) = jacobian(modelTM, Mechano, (uh, θh), du, v, dΩ)
    end
    function jac_termoh(uh, θh)
        jac(dθ, vθ) = jacobian(modelTM, Thermo, dθ, vθ, dΩ)
    end

    jac_ = assemble_matrix(jach(uh, θh), V, V)
    jac_m = assemble_matrix(jac_mech(uh, θh), Vu, Vu)
    jac_t = assemble_matrix(jac_termoh(uh, θh), Vθ, Vθ)


    @test norm(jac_) ≈ 25.6044821500619
    @test jac_[1] ≈ 1.5555555555555538
    @test jac_[end] ≈ 0.33333333333333315
    @test norm(jac_m) ≈ 24.92278198212218
    @test jac_m[1] ≈ 1.5555555555555538
    @test jac_m[end] ≈ 1.8333333333333321
    @test norm(jac_t) ≈ 1.0540925533894592
    @test jac_t[1] ≈ 0.33333333333333304
    @test jac_t[end] ≈ 0.33333333333333315


end

@testset "Assembly Jacobian ElectroMechanics" begin
    modelMR = MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0)
    modelID = IdealDielectric(ε=4.0)
    modelelectro = ElectroMechModel(Mechano=modelMR, Electro=modelID)

    partition = (1, 1, 1)
    pmin = Point(0.0, 0.0, 0.0)
    L = VectorValue(1, 1, 1)
    pmax = pmin + L
    model = CartesianDiscreteModel(pmin, pmax, partition)
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "support0", [1, 2, 3, 4, 5, 6, 7, 8])


    order = 1
    reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
    reffeφ = ReferenceFE(lagrangian, Float64, order)

    #Define test FESpaces
    Vu = TestFESpace(model, reffeu, labels=labels, conformity=:H1)
    Vφ = TestFESpace(model, reffeφ, labels=labels, conformity=:H1)
    V = MultiFieldFESpace([Vu, Vφ])
    U = V
    #Setup integration
    degree = 2 * order
    Ωₕ = Triangulation(model)
    dΩ = Measure(Ωₕ, 2)

    uh = FEFunction(Vu, zeros(Float64, num_free_dofs(Vu)))
    φ(x) = x[1]
    φh = interpolate_everywhere(φ, Vφ)

    function jach(uh, φh)
        jac((du, dφ), (v, vφ)) = jacobian(modelelectro, (uh, φh), (du, dφ), (v, vφ), dΩ)
    end
    function jac_mech(uh, φh)
        jac(du, v) = jacobian(modelelectro, Mechano, (uh, φh), du, v, dΩ)
    end
    function jac_elech(uh, φh)
        jac(dφ, vφ) = jacobian(modelelectro, Electro, (uh, φh), dφ, vφ, dΩ)
    end

    jac_ = assemble_matrix(jach(uh, φh), V, V)
    jac_m = assemble_matrix(jac_mech(uh, φh), Vu, Vu)
    jac_e = assemble_matrix(jac_elech(uh, φh), Vφ, Vφ)
    ≈
    @test norm(jac_) ≈ 18.934585248125135
    @test jac_[1] ≈ 0.7777777777777775
    @test jac_[end] ≈ -1.3333333333333326

    @test norm(jac_m) ≈ 16.420402846143244
    @test jac_m[1] ≈ 0.7777777777777775
    @test jac_m[end] ≈ 2.1111111111111094

    @test norm(jac_e) ≈ 4.216370213557837
    @test jac_e[1] ≈ -1.3333333333333321
    @test jac_e[end] ≈ -1.3333333333333326


end

# @testset "Assembly Jacobian Mechanics" begin

#     pname = "test"
#     meshfile = "test_cubicsphere.msh"
#     simdir = datadir("sims", pname)
#     setupfolder(simdir)
  
#     geomodel = GmshDiscreteModel(datadir("models", meshfile))
  
#     # Constitutive models
#     ∇umacro = TensorValue(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) * 1e-2
  
#     Kmodel = EvolutiveKinematics(Mechano; F=(t) -> ((∇u) -> ∇u + one(∇u) + t * ∇umacro))
#     physmodel_matrix = MooneyRivlin3D(λ=10.0, μ1=1.0, μ2=1.0, Kinematic=Kmodel)
#     physmodel_inclussion = MooneyRivlin3D(λ=50.0, μ1=5.0, μ2=5.0, Kinematic=Kmodel)
  
#     # Setup integration
#     order = 1
#     degree = 2 * order
#     Ω = Triangulation(geomodel)
#     dΩ = Measure(Ω, degree)
#     Ω₁ = Interior(geomodel, tags="Phase0")
#     dΩ₁ = Measure(Ω₁, degree)
#     Ω₂ = Interior(geomodel, tags="Inclusions")
#     dΩ₂ = Measure(Ω₂, degree)
  
#     # Dirichlet conditions 
#     evolu(Λ) = 1.0
#     dir_u_tags = ["Corners"]
#     dir_u_values = [[0.0, 0.0, 0.0]]
#     dir_u_timesteps = [evolu]
#     D_bc = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)
  
#     #  FE spaces
#     reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
#     V = TestFESpace(geomodel, reffeu, D_bc, :H1)
#     U = TrialFESpace(V, D_bc, 1.0)
  
#     #  residual and jacobian function of load factor
#     res(Λ) = (u, v) -> residual(physmodel_matrix, u, v, dΩ₁, Λ) + residual(physmodel_inclussion, u, v, dΩ₂, Λ)
#     jac(Λ) = (u, du, v) -> jacobian(physmodel_matrix, u, du, v, dΩ₁, Λ) + jacobian(physmodel_inclussion, u, du, v, dΩ₂, Λ)
  
#     uh = FEFunction(V, zeros(Float64, num_free_dofs(V)))

#     function jach(uh, φh)
#         jac((du, dφ), (v, vφ)) = jacobian(modelelectro, (uh, φh), (du, dφ), (v, vφ), dΩ)
#     end
#     function jac_mech(uh, φh)
#         jac(du, v) = jacobian(modelelectro, Mechano, (uh, φh), du, v, dΩ)
#     end
#     function jac_elech(uh, φh)
#         jac(dφ, vφ) = jacobian(modelelectro, Electro, (uh, φh), dφ, vφ, dΩ)
#     end

#     jac_ = assemble_matrix(jach(uh, φh), V, V)
#     jac_m = assemble_matrix(jac_mech(uh, φh), Vu, Vu)
#     jac_e = assemble_matrix(jac_elech(uh, φh), Vφ, Vφ)
#     ≈
#     @test norm(jac_) ≈ 18.934585248125135
#     @test jac_[1] ≈ 0.7777777777777775
#     @test jac_[end] ≈ -1.3333333333333326

#     @test norm(jac_m) ≈ 16.420402846143244
#     @test jac_m[1] ≈ 0.7777777777777775
#     @test jac_m[end] ≈ 2.1111111111111094

#     @test norm(jac_e) ≈ 4.216370213557837
#     @test jac_e[1] ≈ -1.3333333333333321
#     @test jac_e[end] ≈ -1.3333333333333326


# end