using Gridap
using ForwardDiff
using JSON
using StaticArrays
using Test
using HyperFEM.PhysicalModels



import Base: -

(-)(A::SMatrix, B::TensorValue) = A - get_array(B)  # NOTE: These functions are required for LinearElasticity to work with ForwardDiff
(-)(A::TensorValue, B::SMatrix) = get_array(A) - B


const ∇u2 = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
const ∇u3 = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
const μParams = [6456.9137547089595, 896.4633794151492,
    1.999999451256222, 1.9999960497608036, 11747.646562400318,
    0.7841068624959612, 1.5386288924587603]


function test_derivatives__(model::PhysicalModel, ∇u; rtol, kwargs...)
  Ψ, ∂Ψu, ∂Ψuu = model()
  ∂Ψu_(F) = TensorValue(ForwardDiff.gradient(Ψ, get_array(F)))
  ∂Ψuu_(F) = TensorValue(ForwardDiff.hessian(Ψ, get_array(F)))

  F, _, _ = get_Kinematics(model.Kinematic)
  @test isapprox(∂Ψu(F(∇u)), ∂Ψu_(F(∇u)), rtol=rtol, kwargs...)
  @test isapprox(∂Ψuu(F(∇u)), ∂Ψuu_(F(∇u)), rtol=rtol, kwargs...)
end

function test_derivatives_2D_(model::PhysicalModel; rtol=1e-14, kwargs...)
  test_derivatives__(model, ∇u2, rtol=rtol, kwargs...)
end

function test_derivatives_3D_(model::PhysicalModel; rtol=1e-14, kwargs...)
  test_derivatives__(model, ∇u3, rtol=rtol, kwargs...)
end



@testset "Iso+Aniso" begin
  N = VectorValue(1.0, 2.0, 3.0)
  model1 = MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0)
  model2 = TransverseIsotropy3D(μ=μParams[5], α=μParams[6], β=μParams[7])
  model=model1+model2

  Ψ, ∂Ψu, ∂Ψuu = model()
  Ψ1, ∂Ψu1, ∂Ψuu1 = model1()
  Ψ2, ∂Ψu2, ∂Ψuu2 = model2()

  F, _, _ = get_Kinematics(model.Kinematic)
  @test Ψ(F(∇u3), N) == Ψ1(F(∇u3)) + Ψ2(F(∇u3), N)
  @test norm(∂Ψu(F(∇u3), N)) == norm(∂Ψu1(F(∇u3))+∂Ψu2(F(∇u3), N) )
  @test norm(∂Ψuu(F(∇u3), N)) == norm(∂Ψuu1(F(∇u3))+∂Ψuu2(F(∇u3), N))
end


@testset "Iso+MultiAniso" begin
  N1 = VectorValue(0.0, 0.0, 1.0)
  N2 = VectorValue(0.0, 1.0, 0.0)

  model1 = MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0)
  model2 = TransverseIsotropy3D(μ=μParams[5], α=μParams[6], β=μParams[7])
  model3 = TransverseIsotropy3D(μ=μParams[5]*2, α=μParams[6], β=μParams[7])

  model=model1+[model2 model3]
 
  Ψ, ∂Ψu, ∂Ψuu = model()
  Ψ1, ∂Ψu1, ∂Ψuu1 = model1()
  Ψ2, ∂Ψu2, ∂Ψuu2 = model2()
  Ψ3, ∂Ψu3, ∂Ψuu3 = model3()

  F, _, _ = get_Kinematics(model.Kinematic)
  @test Ψ(F(∇u3), [N1,N2]) == Ψ1(F(∇u3)) + Ψ2(F(∇u3), [N1]) + Ψ3(F(∇u3), [N2])
  @test isapprox(norm(∂Ψu(F(∇u3), [N1,N2])), norm(∂Ψu1(F(∇u3))+∂Ψu2(F(∇u3), N1)+∂Ψu3(F(∇u3), N2)), rtol=1e-14)
  @test isapprox(norm(∂Ψuu(F(∇u3), [N1,N2])), norm(∂Ψuu1(F(∇u3))+∂Ψuu2(F(∇u3), N1)+∂Ψuu3(F(∇u3), N2)), rtol=1e-14)
end



@testset "NonlinearMooneyRivlin_CV" begin
  model = NonlinearMooneyRivlin_CV(λ=3.0, μ1=1.0, μ2=1.0, α=2.0, β=1.0, γ=6.0)
  test_derivatives_3D_(model, rtol=1e-13)
end


@testset "NonlinearNeoHookean_CV" begin
  model = NonlinearNeoHookean_CV(λ=3.0, μ=1.0, α=2.0, γ=6.0)
  test_derivatives_3D_(model, rtol=1e-13)
end


@testset "IncompressibleNeoHookean3D_2dP" begin
  Ce = TensorValue(0.01 + 1.0, 0.02, 0.03, 0.04, 0.05 + 1.0, 0.06, 0.07, 0.08, 0.09 + 1.0)
  model = IncompressibleNeoHookean3D_2dP(μ=1.0, τ=1.0, Δt=1.0)
  Ψ, Se, ∂Se = model()
  F, H, _ = get_Kinematics(model.Kinematic)



  # Se_(Ce) =2*TensorValue(ForwardDiff.gradient(x -> Ψ(x), get_array(Ce)))
  # ∂Se_(Ce) =2*TensorValue(ForwardDiff.hessian(x -> Ψ(x), get_array(Ce)))

  #  norm(Se_(Ce)) -norm(Se(Ce))
  #  norm(∂Se_(Ce)) -norm(∂Se(Ce))
  @test (Ψ(Ce)) == 1.5040930711508358
  @test norm(Se(Ce)) == 0.12632997589595116
  @test norm(∂Se(Ce)) == 2.616897862779383

end


@testset "LinearElasticity2D" begin
  model = LinearElasticity2D(λ=3.0, μ=1.0)
  test_derivatives_2D_(model)
end


@testset "LinearElasticity3D" begin
  model = LinearElasticity3D(λ=3.0, μ=1.0)
  test_derivatives_3D_(model)
end


@testset "NeoHookean3D" begin
  model = NeoHookean3D(λ=3.0, μ=1.0)
  test_derivatives_3D_(model, rtol=1e-13)
end


@testset "MooneyRivlin2D" begin
  model = MooneyRivlin2D(λ=3.0, μ1=1.0, μ2=2.0)
  test_derivatives_2D_(model)
end

@testset "MooneyRivlin3D" begin
  model = MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0)
  test_derivatives_3D_(model, rtol=1e-13)
end


@testset "NonlinearMooneyRivlin2D" begin
  model = NonlinearMooneyRivlin2D(λ=(μParams[1] + μParams[2]) * 1e2, μ1=μParams[1], μ2=μParams[2], α=μParams[3], β=μParams[4])
  test_derivatives_2D_(model)
end


@testset "Yeoh3D" begin
  model = Yeoh3D(λ=3.0, C10=1.0, C20=1.0, C30=1.0)
  test_derivatives_3D_(model)
end


@testset "NonlinearMooneyRivlin2D_CV" begin
  model = NonlinearMooneyRivlin2D_CV(λ=(μParams[1] + μParams[2]) * 1e2, μ1=μParams[1], μ2=μParams[2], α=μParams[3], β=μParams[4], γ=μParams[4])
  test_derivatives_2D_(model)
end


@testset "NonlinearMooneyRivlin3D" begin
  model = NonlinearMooneyRivlin3D(λ=(μParams[1] + μParams[2]) * 1e2, μ1=μParams[1], μ2=μParams[2], α=μParams[3], β=μParams[4])
  test_derivatives_3D_(model, rtol=1e-13)
end


@testset "IncompressibleNeoHookean2D" begin
  model = IncompressibleNeoHookean2D(λ=(μParams[1] + μParams[2]) * 1e2, μ=μParams[1])
  test_derivatives_2D_(model)
end

@testset "IncompressibleNeoHookean2D_CV" begin
  model = IncompressibleNeoHookean2D_CV(λ=(μParams[1] + μParams[2]) * 1e2, μ=μParams[1], γ=3.0)
  test_derivatives_2D_(model)
end


@testset "NonlinearIncompressibleMooneyRivlin2D_CV" begin
  model = NonlinearIncompressibleMooneyRivlin2D_CV(λ=(μParams[1] + μParams[2]) * 1e2, μ=μParams[1], α=μParams[3], γ=3.0)
  test_derivatives_2D_(model)
end


@testset "EightChain" begin
  model = EightChain(μ=μParams[1], N=μParams[2])
  test_derivatives_3D_(model, rtol=1e-13)
end


@testset "TransverseIsotropy2D" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
  ∇u0 = TensorValue(1.0, 2.0, 3.0, 4.0) * 0.0

  N = VectorValue(1.0, 2.0) / sqrt(5.0)
  model = TransverseIsotropy2D(μ=μParams[5], α=μParams[6], β=μParams[7])
  Ψ, ∂Ψu, ∂Ψuu = model()

  F, _, J_ = get_Kinematics(model.Kinematic)

  # ∂Ψu_(F,N) =TensorValue(ForwardDiff.gradient(x -> Ψ(x,get_array(N)), get_array(F)))
  # ∂Ψuu_(F,N) =TensorValue(ForwardDiff.hessian(x -> Ψ(x,get_array(N)), get_array(F)))

  # norm(∂Ψu_(F(∇u),N)) - norm(∂Ψu(F(∇u0),N))
  # norm(∂Ψuu_(F(∇u),N)) - norm(∂Ψuu(F(∇u),N))


  @test Ψ(F(∇u), N) == 0.27292220826242186
  @test norm(∂Ψu(F(∇u), N)) == 100.64088114687468
  @test norm(∂Ψuu(F(∇u), N)) == 46792.35008576098
end





@testset "TransverseIsotropy3D" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  N = VectorValue(1.0, 2.0, 3.0)
  model = TransverseIsotropy3D(μ=μParams[5], α=μParams[6], β=μParams[7])

  Ψ, ∂Ψu, ∂Ψuu = model()
  F, _, _ = get_Kinematics(model.Kinematic)
  @test Ψ(F(∇u), N) == 269927.3350807581
  @test norm(∂Ψu(F(∇u), N)) == 947447.8711645481
  @test norm(∂Ψuu(F(∇u), N)) == 3.8258646319087776e6
end



  



@testset "TermoElectroMech" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  ∇φ = VectorValue(1.0, 2.0, 3.0)
  θt = 3.4 - 1.0
  modelMR = MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0)
  modelID = IdealDielectric(ε=4.0)
  modelT = ThermalModel(Cv=1.0, θr=1.0, α=2.0)
  f(δθ::Float64)::Float64 = (δθ + 1.0) / 1.0
  df(δθ::Float64)::Float64 = 1.0
  modelTEM = ThermoElectroMechModel(modelT, modelID, modelMR, fθ=f, dfdθ=df)
  Ψ, ∂Ψu, ∂Ψφ, ∂Ψθ, ∂Ψuu, ∂Ψφφ, ∂Ψθθ, ∂Ψφu, ∂Ψuθ, ∂Ψφθ = modelTEM()
  F, _, _ = get_Kinematics(modelMR.Kinematic)
  E = get_Kinematics(modelID.Kinematic)

  @test (Ψ(F(∇u), E(∇φ), θt)) == -95.74389746463744
  @test norm(∂Ψu(F(∇u), E(∇φ), θt)) == 185.1315441384458
  @test norm(∂Ψφ(F(∇u), E(∇φ), θt)) == 50.00690431860902
  @test norm(∂Ψθ(F(∇u), E(∇φ), θt)) == 28.91912594899454
  @test norm(∂Ψuu(F(∇u), E(∇φ), θt)) == 429.9957659123366
  @test norm(∂Ψφφ(F(∇u), E(∇φ), θt)) == 23.679055285771508
  @test norm(∂Ψθθ(F(∇u), E(∇φ), θt)) == 0.29411764705882354
  @test norm(∂Ψφu(F(∇u), E(∇φ), θt)) == 132.7243219000811
  @test norm(∂Ψuθ(F(∇u), E(∇φ), θt)) == 58.281073490042175
  @test norm(∂Ψφθ(F(∇u), E(∇φ), θt)) == 14.707913034885005
end


@testset "TermoMech" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  θt = 3.4 - 1.0
  modelMR = MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0)
  modelT = ThermalModel(Cv=1.0, θr=1.0, α=2.0)
  f(δθ::Float64)::Float64 = (δθ + 1.0) / 1.0
  df(δθ::Float64)::Float64 = 1.0
  modelTM = ThermoMechModel(modelT, modelMR, fθ=f, dfdθ=df)
  Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ = modelTM()
  F, _, _ = get_Kinematics(modelMR.Kinematic)

  @test (Ψ(F(∇u), θt)) == -2.190116215314799
  @test norm(∂Ψu(F(∇u), θt)) == 50.34457217400186
  @test norm(∂Ψθ(F(∇u), θt)) == 1.4033079344878807
  @test norm(∂Ψuu(F(∇u), θt)) == 132.85408867418602
  @test norm(∂Ψθθ(F(∇u), θt)) == 0.29411764705882354
  @test norm(∂Ψuθ(F(∇u), θt)) == 21.074087978716364
end


@testset "ThermoMech_EntropicPolyconvex" begin
  ∇u = 1e-1 * TensorValue(1, 2, 3, 4, 5, 6, 7, 8, 9)
  θt = 21.6
  modmec = MooneyRivlin3D(λ=10.0, μ1=1.0, μ2=1.0, ρ=1.0)
  modterm = ThermalModel(Cv=3.4, θr=2.2, α=1.2, κ=1.0)
  β = 0.7
  G(x) = x * (log(x) - 1.0) - 4 / 3 * x^(3 / 2) + 2 * x + 1 / 3
  γ₁ = 0.5
  γ₂ = 0.5
  γ₃ = 0.5
  s(I1, I2, I3) = 1 / 3 * ((I1 / 3.0)^γ₁ + (I2 / 3.0)^γ₂ + I3^γ₃)
  ϕ(x) = 2.0 * (x + 1.0) * log(x + 1.0) - 2.0 * x * (1 + log(2)) + 2.0 * (1 - log(2))
  consmodel = ThermoMech_EntropicPolyconvex(modterm, modmec, β=β, G=G, ϕ=ϕ, s=s)

  Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ = consmodel()
  F, _, _ = get_Kinematics(modmec.Kinematic)

  @test (Ψ(F(∇u), θt)) == -129.4022076861008
  @test norm(∂Ψu(F(∇u), θt)) == 437.9269386687991
  @test norm(∂Ψθ(F(∇u), θt)) == 13.97666807099424
  @test norm(∂Ψuu(F(∇u), θt)) == 2066.7910102392775
  @test norm(∂Ψθθ(F(∇u), θt)) == 0.46689338540182707
  @test norm(∂Ψuθ(F(∇u), θt)) == 14.243050132210923
end


@testset "ThermoElectroMech_Bonet" begin

  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  ∇φ = VectorValue(1.0, 2.0, 3.0)
  θt = 3.4 - 1.0
  modelMR = MooneyRivlin3D(λ=0.0, μ1=0.5, μ2=0.5)
  modelID = IdealDielectric(ε=1.0)
  modelT = ThermalModel(Cv=17.385, θr=293.0, α=0.00156331, γv=2.0, γd=2.0)

  modelTEM = ThermoElectroMech_Bonet(modelT, modelID, modelMR)
  Ψ, ∂Ψu, ∂ΨE, ∂Ψθ, ∂ΨFF, ∂ΨEE, ∂2Ψθθ, ∂ΨEF, ∂ΨFθ, ∂ΨEθ, η = modelTEM()

  F, _, _ = get_Kinematics(modelMR.Kinematic)
  E = get_Kinematics(modelID.Kinematic)

  ∂Ψ_∂F(F, E, θ) = TensorValue(ForwardDiff.gradient(F -> Ψ(F, get_array(E), θ), get_array(F)))
  ∂Ψ_∂E(F, E, θ) = VectorValue(ForwardDiff.gradient(E -> Ψ(get_array(F), E, θ), get_array(E)))
  ∂Ψ_∂θ(F, E, θ) = ForwardDiff.derivative(θ -> Ψ(get_array(F), get_array(E), θ), θ)


  ∂2Ψ_∂2E(F, E, θ) = TensorValue(ForwardDiff.hessian(E -> Ψ(get_array(F), E, θ), get_array(E)))
  ∂2Ψ∂2θ(F, E, θ) = ForwardDiff.derivative(θ -> ∂Ψ_∂θ(get_array(F), get_array(E), θ), θ)
  ∂2Ψ_∂2Eθ(F, E, θ) = VectorValue(ForwardDiff.derivative(θ -> get_array(∂Ψ_∂E(get_array(F), get_array(E), θ)), θ))
  ∂2Ψ_∂2F(F, E, θ) = TensorValue(ForwardDiff.hessian(F -> Ψ(F, get_array(E), θ), get_array(F)))
  ∂2Ψ_∂2Fθ(F, E, θ) = TensorValue(ForwardDiff.derivative(θ -> get_array(∂Ψ_∂F(get_array(F), get_array(E), θ)), θ))
  ∂2Ψ_∂EF(F, E, θ) = TensorValue(ForwardDiff.jacobian(F -> get_array(∂Ψ_∂E(F, get_array(E), θ)), get_array(F)))


  @test isapprox(∂Ψu(F(∇u), E(∇φ), θt), ∂Ψ_∂F(F(∇u), E(∇φ), θt); rtol=1e-14)
  @test isapprox(∂ΨE(F(∇u), E(∇φ), θt), ∂Ψ_∂E(F(∇u), E(∇φ), θt); rtol=1e-14)
  @test isapprox(∂Ψθ(F(∇u), E(∇φ), θt), ∂Ψ_∂θ(F(∇u), E(∇φ), θt); rtol=1e-14)
  @test isapprox(∂ΨEE(F(∇u), E(∇φ), θt), ∂2Ψ_∂2E(F(∇u), E(∇φ), θt); rtol=1e-14)
  @test isapprox(∂2Ψθθ(F(∇u), E(∇φ), θt), ∂2Ψ∂2θ(F(∇u), E(∇φ), θt); rtol=1e-14)
  @test isapprox(∂ΨEθ(F(∇u), E(∇φ), θt), ∂2Ψ_∂2Eθ(F(∇u), E(∇φ), θt); rtol=1e-14)
  @test isapprox(∂ΨFF(F(∇u), E(∇φ), θt), ∂2Ψ_∂2F(F(∇u), E(∇φ), θt); rtol=1e-14)
  @test isapprox(∂ΨFθ(F(∇u), E(∇φ), θt), ∂2Ψ_∂2Fθ(F(∇u), E(∇φ), θt); rtol=1e-14)
  @test isapprox(∂ΨEF(F(∇u), E(∇φ), θt), ∂2Ψ_∂EF(F(∇u), E(∇φ), θt); rtol=1e-14)

 end


@testset "VolumetricEnergy" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  model = VolumetricEnergy(λ=0.0 )
 
   Ψ, ∂Ψu, ∂Ψuu= model()
  F, _, _ = get_Kinematics(model.Kinematic)
 
  ∂Ψu_(F) =TensorValue(ForwardDiff.gradient(x -> Ψ(x), get_array(F)))
  ∂Ψuu_(F) =TensorValue(ForwardDiff.hessian(x -> Ψ(x), get_array(F)))

  @test isapprox(∂Ψu(F(∇u)), ∂Ψu_(F(∇u)); rtol=1e-14)
  @test isapprox(∂Ψuu(F(∇u)), ∂Ψuu_(F(∇u)); rtol=1e-14)
 
end




@testset "ThermoElectroMech_Govindjee" begin

  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  ∇φ = VectorValue(1.0, 2.0, 3.0)
  θt = 3.4 - 1.0
  modelMR = MooneyRivlin3D(λ=5.0, μ1=0.5, μ2=0.5)
  modelID = IdealDielectric(ε=1.0)
  modelT = ThermalModel(Cv=17.385, θr=293.0, α=0.00156331)
  f(δθ) = (δθ + 293.0) / 293.0
  df(δθ) = 293.0
  g(δθ) = -0.33 * ((δθ + 293.0) / 293.0)^3
  dg(δθ) = -(3 * 0.33 / 293.0) * ((δθ + 293.0) / 293.0)^2

  modelTEM = ThermoElectroMech_Govindjee(modelT, modelID, modelMR, fθ=f, dfdθ=df, gθ=g, dgdθ=dg, β=0.0)
  Ψ, ∂Ψu, ∂ΨE, ∂Ψθ, ∂ΨFF, ∂ΨEE, ∂2Ψθθ, ∂ΨEF, ∂ΨFθ, ∂ΨEθ, η = modelTEM()
  F, _, _ = get_Kinematics(modelMR.Kinematic)
  E = get_Kinematics(modelID.Kinematic)

  @test Ψ(F(∇u), E(∇φ), θt) == -7.104365408674424
  @test norm(∂Ψu(F(∇u), E(∇φ), θt)) == 11.921289845756304
  @test norm(∂ΨE(F(∇u), E(∇φ), θt)) == 3.7068519469562706
  @test ∂Ψθ(F(∇u), E(∇φ), θt) == -0.1649382571669807
  @test norm(∂ΨFF(F(∇u), E(∇φ), θt)) == 38.03251633659781
  @test norm(∂ΨEE(F(∇u), E(∇φ), θt)) == 1.7552526672898596
  @test norm(∂2Ψθθ(F(∇u), E(∇φ), θt)) == 0.05869247142552643
  @test norm(∂ΨEF(F(∇u), E(∇φ), θt)) == 9.838429667814548
  @test norm(∂ΨFθ(F(∇u), E(∇φ), θt)) == 0.04069091555160856
  @test norm(∂ΨEθ(F(∇u), E(∇φ), θt)) == 0.012345048484459126



end


@testset "ThermoElectroMech_PINNS" begin

  function ExtractingInfo(data_filename)
    data_dict = open(data_filename, "r") do file
      JSON.parse(file)
    end
    weights_ = data_dict["weights"]
    biases_ = data_dict["biases"]
    Scaling = data_dict["Scaling"]
    ϵ = vcat(Scaling["ϵₓ"], Scaling["ϵθ"])
    β = vcat(Scaling["βₓ"], Scaling["βθ"])
    n_layers = size(weights_, 1)
    Weights = Vector{Matrix{Float64}}(undef, n_layers)
    Biases = Vector{Any}(undef, n_layers)
    for i in 1:n_layers
      Weights[i] = hcat(weights_[i]...)  # Concatenate weights horizontally
      if length(biases_[i]) == 1 && isa(biases_[i][1], Float64)
        Biases[i] = biases_[i][1]  # Convert 1-element Vector{Any} to Float64
      else
        Biases[i] = biases_[i]  # Assign directly if it's a vector
      end
    end
    return n_layers, Weights, Biases, ϵ, β
  end

  BASE_FOLDER = dirname(dirname(pathof(HyperFEM)))
  data_filename = joinpath(BASE_FOLDER, "test/models/test_NN_TEM.json")
  n_layers, Weights, Biases, ϵ, β = ExtractingInfo(data_filename)

  model = ThermoElectroMech_PINNs(; W=Weights, b=Biases, ϵ=ϵ, β=β, nLayer=n_layers)

  Ψ, ∂ΨF, ∂ΨE, ∂Ψθ, ∂ΨFF, ∂ΨEE, ∂Ψθθ, ∂ΨEF, ∂ΨFθ, ∂ΨEθ = model()

  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  ∇φ = VectorValue(1.0, 2.0, 3.0)
  θt = 3.4 - 1.0
  Kinematic_mec = Kinematics(Mechano)
  Kinematic_elec = Kinematics(Electro)
  F, _, _ = get_Kinematics(Kinematic_mec)
  E = get_Kinematics(Kinematic_elec)

  @test isapprox(Ψ(F(∇u), E(∇φ), θt), 34.24573625846419, atol=1e-12)
  @test norm(∂ΨF(F(∇u), E(∇φ), θt)) == 12.190784442767743
  @test norm(∂ΨE(F(∇u), E(∇φ), θt)) == 3.890788259241063
  @test ∂Ψθ(F(∇u), E(∇φ), θt) == -0.1756808680132173
  @test norm(∂ΨFF(F(∇u), E(∇φ), θt)) == 41.75134321258517
  @test norm(∂ΨEE(F(∇u), E(∇φ), θt)) == 1.9388101847663917
  @test norm(∂Ψθθ(F(∇u), E(∇φ), θt)) == 0.05854786347086507
  @test norm(∂ΨEF(F(∇u), E(∇φ), θt)) == 10.455220025096452
  @test norm(∂ΨFθ(F(∇u), E(∇φ), θt)) == 0.059252287541736004
  @test norm(∂ΨEθ(F(∇u), E(∇φ), θt)) == 0.023111702806623537

end

 

@testset "IdealMagnetic2D" begin

  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
  ∇φ = VectorValue(1.0, 2.0)

  modelMR = MooneyRivlin2D(λ=3.0, μ1=1.0, μ2=2.0)

  modelID = IdealMagnetic2D(μ=1.2566e-6, χe=0.0)
  Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = modelID()
  F, _, _ = get_Kinematics(modelMR.Kinematic)
  H0 = get_Kinematics(modelID.Kinematic)


  # ∂Ψφ_(H) =VectorValue(ForwardDiff.gradient(x -> Ψ(get_array( F(∇u)), x,get_array(N) ), get_array(H)))
  # ∂Ψφφ_(H) =TensorValue(ForwardDiff.jacobian(x -> ∂Ψφ(get_array( F(∇u)), x,get_array(N) ), get_array(H)))

  # norm(∂Ψφ_(H0(∇φ))) - norm(∂Ψφ(F(∇u), H0(∇φ), N))
  # norm(∂Ψφφ_(H0(∇φ))) -norm(∂Ψφφ(F(∇u), H0(∇φ), N))

  @test Ψ(F(∇u), H0(∇φ)) == -3.123376791098092e-6
  @test norm(∂Ψφ(F(∇u), H0(∇φ))) == 2.793633631007779e-6
  @test norm(∂Ψφφ(F(∇u), H0(∇φ))) == 1.7771608829110207e-6
end




@testset "IdealMagnetic" begin


  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  ∇φ = VectorValue(1.0, 2.0, 3.0)

  modelMR = MooneyRivlin2D(λ=3.0, μ1=1.0, μ2=2.0)

  modelID = IdealMagnetic(μ=1.2566e-6, χe=0.0)
  Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = modelID()
  F, _, _ = get_Kinematics(modelMR.Kinematic)
  H0 = get_Kinematics(modelID.Kinematic)


  # ∂Ψu_(F) =TensorValue(ForwardDiff.gradient(x -> Ψ(x, get_array( H0(∇φ)),get_array(N) ), get_array(F)))
  # ∂Ψφ_(H) =VectorValue(ForwardDiff.gradient(x -> Ψ(get_array( F(∇u)), x,get_array(N) ), get_array(H)))
  # ∂Ψuu_(F) =TensorValue(ForwardDiff.hessian(x -> Ψ(x, get_array( H0(∇φ)),get_array(N) ), get_array(F)))
  # ∂Ψφu_(H) =TensorValue(ForwardDiff.jacobian(x -> ∂Ψφ(x, get_array( H0(∇φ)),get_array(N) ), get_array(F(∇u))))
  # ∂Ψφφ_(H) =TensorValue(ForwardDiff.jacobian(x -> ∂Ψφ(get_array( F(∇u)), x,get_array(N) ), get_array(H)))

  # norm(∂Ψu_(F(∇u))) -   norm(∂Ψu(F(∇u), H0(∇φ), N))
  # norm(∂Ψφ_(H0(∇φ))) - norm(∂Ψφ(F(∇u), H0(∇φ), N))
  # norm(∂Ψuu_(F(∇u))) -norm(∂Ψuu(F(∇u), H0(∇φ), N))
  # norm(∂Ψφu_(H0(∇φ))-∂Ψφu(F(∇u), H0(∇φ), N))  
  # norm(∂Ψφφ_(H0(∇φ))) -norm(∂Ψφφ(F(∇u), H0(∇φ), N))

  @test Ψ(F(∇u), H0(∇φ)) == -8.644094229257268e-6
  @test norm(∂Ψu(F(∇u), H0(∇φ))) == 1.4898943079174831e-5
  @test norm(∂Ψφ(F(∇u), H0(∇φ))) == 4.620490879909124e-6
  @test norm(∂Ψuu(F(∇u), H0(∇φ))) == 4.193626521492582e-5
  @test norm(∂Ψφu(F(∇u), H0(∇φ))) == 1.2263336977914849e-5
  @test norm(∂Ψφφ(F(∇u), H0(∇φ))) == 2.1878750641250348e-6
end



@testset "HardMagnetic_SoftMaterial3D" begin

  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  ∇φ = VectorValue(1.0, 2.0, 3.0)
  N = VectorValue(0.0, 0.0, 1.0)

  modelMR = MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0)
  modelID = HardMagnetic(μ=1.2566e-6, αr=40e-3, χe=0.0, χr=8.0)
  modelmagneto = MagnetoMechModel(modelID, modelMR)
  Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = modelmagneto()
  F, _, _ = get_Kinematics(modelMR.Kinematic)
  H0 = get_Kinematics(modelID.Kinematic)


  # ∂Ψu_(F) =TensorValue(ForwardDiff.gradient(x -> Ψ(x, get_array( H0(∇φ)),get_array(N) ), get_array(F)))
  # ∂Ψφ_(H) =VectorValue(ForwardDiff.gradient(x -> Ψ(get_array( F(∇u)), x,get_array(N) ), get_array(H)))
  # ∂Ψuu_(F) =TensorValue(ForwardDiff.hessian(x -> Ψ(x, get_array( H0(∇φ)),get_array(N) ), get_array(F)))
  # ∂Ψφu_(H) =TensorValue(ForwardDiff.jacobian(x -> ∂Ψφ(x, get_array( H0(∇φ)),get_array(N) ), get_array(F(∇u))))
  # ∂Ψφφ_(H) =TensorValue(ForwardDiff.jacobian(x -> ∂Ψφ(get_array( F(∇u)), x,get_array(N) ), get_array(H)))

  # norm(∂Ψu_(F(∇u))) 
  # norm(∂Ψφ_(H0(∇φ))) 
  # norm(∂Ψuu_(F(∇u))) 
  # norm(∂Ψφu_(H0(∇φ))) 
  # norm(∂Ψφφ_(H0(∇φ))) 


  @test Ψ(F(∇u), H0(∇φ), N) == 0.001589466682574581
  @test norm(∂Ψu(F(∇u), H0(∇φ), N)) == 0.24833301570214883
  @test norm(∂Ψφ(F(∇u), H0(∇φ), N)) == 4.660348298920368e-6
  @test norm(∂Ψuu(F(∇u), H0(∇φ), N)) == 30.36786063436432
  @test norm(∂Ψφu(F(∇u), H0(∇φ), N)) == 1.2369035467980284e-5
  @test norm(∂Ψφφ(F(∇u), H0(∇φ), N)) == 2.1878750641250348e-6
end


@testset "Magnetic3D" begin
  ∇φ = VectorValue(1.0, 2.0, 3.0)
  a=40e-3 
  ra=Ref(a)
  modelID = Magnetic(μ=1.2566e-6, αr=ra ,χe=0.0)
  Ψ, ∂Ψφ, ∂Ψφφ = modelID()
  H0 = get_Kinematics(modelID.Kinematic)
  N = VectorValue(0.0, 0.0, 1.0)

  #  ∂Ψφ_(H) =VectorValue(ForwardDiff.gradient(x -> Ψ( x,get_array(N) ), get_array(H)))
  #  ∂Ψφφ_(H) =TensorValue(ForwardDiff.jacobian(x -> ∂Ψφ( x,get_array(N) ), get_array(H)))

  #  norm(∂Ψφ_(H0(∇φ))) - norm(∂Ψφ( H0(∇φ), N))
  #  norm(∂Ψφφ_(H0(∇φ))) -norm(∂Ψφφ(H0(∇φ), N))

  @test Ψ(H0(∇φ),N) == -8.64641328e-6
  @test norm(∂Ψφ(H0(∇φ),N)) == 4.6615625980239715e-6
  @test norm(∂Ψφφ(H0(∇φ),N)) == 2.1764950447910513e-6
end



@testset "Magnetic2D" begin
  ∇φ = VectorValue(1.0, 2.0)
  a=40e-3 
  ra=Ref(a)
  modelID = Magnetic(μ=1.2566e-6, αr=ra ,χe=0.0)
  Ψ, ∂Ψφ, ∂Ψφφ = modelID()
  H0 = get_Kinematics(modelID.Kinematic)
  N = VectorValue(0.0, 0.0)

  #  ∂Ψφ_(H) =VectorValue(ForwardDiff.gradient(x -> Ψ( x,get_array(N) ), get_array(H)))
  #  ∂Ψφφ_(H) =TensorValue(ForwardDiff.jacobian(x -> ∂Ψφ( x,get_array(N) ), get_array(H)))

  #  norm(∂Ψφ_(H0(∇φ))) - norm(∂Ψφ( H0(∇φ), N))
  #  norm(∂Ψφφ_(H0(∇φ))) -norm(∂Ψφφ(H0(∇φ), N))

  @test Ψ(H0(∇φ),N) == -3.1415e-6
  @test norm(∂Ψφ(H0(∇φ),N)) == 2.8098430205262357e-6
  @test norm(∂Ψφφ(H0(∇φ),N)) == 1.7771007624780312e-6
end




@testset "IdealMagnetic_SoftMaterial2D" begin

  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
  ∇φ = VectorValue(1.0, 2.0)
  N = VectorValue(0.0, 1.0)

  modelMR = MooneyRivlin2D(λ=3.0, μ1=1.0, μ2=2.0)
  modelID = IdealMagnetic2D(μ=1.2566e-6, χe=0.0)
  modelmagneto = MagnetoMechModel(modelID, modelMR)
  Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = modelmagneto()
  F, _, _ = get_Kinematics(modelMR.Kinematic)
  H0 = get_Kinematics(modelID.Kinematic)


  # ∂Ψu_(F) =TensorValue(ForwardDiff.gradient(x -> Ψ(x, get_array( H0(∇φ)),get_array(N) ), get_array(F)))
  # ∂Ψφ_(H) =VectorValue(ForwardDiff.gradient(x -> Ψ(get_array( F(∇u)), x,get_array(N) ), get_array(H)))
  # ∂Ψuu_(F) =TensorValue(ForwardDiff.hessian(x -> Ψ(x, get_array( H0(∇φ)),get_array(N) ), get_array(F)))
  # ∂Ψφu_(H) =TensorValue(ForwardDiff.jacobian(x -> ∂Ψφ(x, get_array( H0(∇φ)),get_array(N) ), get_array(F(∇u))))
  # ∂Ψφφ_(H) =TensorValue(ForwardDiff.jacobian(x -> ∂Ψφ(get_array( F(∇u)), x,get_array(N) ), get_array(H)))

  # norm(∂Ψu_(F(∇u))) -   norm(∂Ψu(F(∇u), H0(∇φ), N))
  # norm(∂Ψφ_(H0(∇φ))) - norm(∂Ψφ(F(∇u), H0(∇φ), N))
  # norm(∂Ψuu_(F(∇u))) -norm(∂Ψuu(F(∇u), H0(∇φ), N))
  # norm(∂Ψφu_(H0(∇φ))-∂Ψφu(F(∇u), H0(∇φ), N))  
  # norm(∂Ψφφ_(H0(∇φ))) -norm(∂Ψφφ(F(∇u), H0(∇φ), N))





  @test Ψ(F(∇u), H0(∇φ), N) == 4.000172569336671
  @test norm(∂Ψu(F(∇u), H0(∇φ), N)) == 0.07482084634773895
  @test norm(∂Ψφ(F(∇u), H0(∇φ), N)) == 2.793633631007779e-6
  @test norm(∂Ψuu(F(∇u), H0(∇φ), N)) == 21.74472389462642
  @test norm(∂Ψφu(F(∇u), H0(∇φ), N)) == 5.589596497314291e-6
  @test norm(∂Ψφφ(F(∇u), H0(∇φ), N)) == 1.7771608829110207e-6
end





@testset "HardMagnetic_SoftMaterial2D" begin

  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
  ∇φ = VectorValue(1.0, 2.0)
  N = VectorValue(0.0, 1.0)

  modelMR = MooneyRivlin2D(λ=3.0, μ1=1.0, μ2=2.0)
  modelID = HardMagnetic2D(μ=1.2566e-6, αr=40e-3, χe=0.0, χr=8.0)
  modelmagneto = MagnetoMechModel(modelID, modelMR)
  Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = modelmagneto()
  F, _, _ = get_Kinematics(modelMR.Kinematic)
  H0 = get_Kinematics(modelID.Kinematic)


  # ∂Ψu_(F) =TensorValue(ForwardDiff.gradient(x -> Ψ(x, get_array( H0(∇φ)),get_array(N) ), get_array(F)))
  # ∂Ψφ_(H) =VectorValue(ForwardDiff.gradient(x -> Ψ(get_array( F(∇u)), x,get_array(N) ), get_array(H)))
  # ∂Ψuu_(F) =TensorValue(ForwardDiff.hessian(x -> Ψ(x, get_array( H0(∇φ)),get_array(N) ), get_array(F)))
  # ∂Ψφu_(H) =TensorValue(ForwardDiff.jacobian(x -> ∂Ψφ(x, get_array( H0(∇φ)),get_array(N) ), get_array(F(∇u))))
  # ∂Ψφφ_(H) =TensorValue(ForwardDiff.jacobian(x -> ∂Ψφ(get_array( F(∇u)), x,get_array(N) ), get_array(H)))

  # norm(∂Ψu_(F(∇u))) -   norm(∂Ψu(F(∇u), H0(∇φ), N))
  # norm(∂Ψφ_(H0(∇φ))) - norm(∂Ψφ(F(∇u), H0(∇φ), N))
  # norm(∂Ψuu_(F(∇u))) -norm(∂Ψuu(F(∇u), H0(∇φ), N))
  # norm(∂Ψφu_(H0(∇φ))-∂Ψφu(F(∇u), H0(∇φ), N))  
  # norm(∂Ψφφ_(H0(∇φ))) -norm(∂Ψφφ(F(∇u), H0(∇φ), N))


  @test Ψ(F(∇u), H0(∇φ), N) == 4.000172469501178
  @test norm(∂Ψu(F(∇u), H0(∇φ), N)) == 0.07482089298212842
  @test norm(∂Ψφ(F(∇u), H0(∇φ), N)) == 2.8384487487963508e-6
  @test norm(∂Ψuu(F(∇u), H0(∇φ), N)) == 21.744723980670503
  @test norm(∂Ψφu(F(∇u), H0(∇φ), N)) == 5.679235813302821e-6
  @test norm(∂Ψφφ(F(∇u), H0(∇φ), N)) == 1.7771608829110207e-6
end









@testset "ARAP2D" begin
  model = ARAP2D(μ=μParams[1])
  test_derivatives_2D_(model, rtol=1e-13)
end




@testset "ARAP2D_regularized" begin
  model = ARAP2D_regularized(μ=μParams[1])
  test_derivatives_2D_(model, rtol=1e-13)
end


@testset "HessianRegularization" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
  ∇u0 = TensorValue(1.0, 2.0, 3.0, 4.0) * 0.0

  model = ARAP2D_regularized(μ=μParams[1])
  modelreg = HessianRegularization(model)

  Ψ, ∂Ψu, ∂Ψuu = modelreg()
  F, _, J_ = get_Kinematics(modelreg.Kinematic)

  # ∂Ψu_(F) =TensorValue(ForwardDiff.gradient(x -> Ψ(x), get_array(F)))
  # ∂Ψuu_(F) =TensorValue(ForwardDiff.hessian(x -> Ψ(x), get_array(F)))
  # norm(∂Ψu_(F(∇u))) - norm(∂Ψu(F(∇u)))
  #  norm(∂Ψuu_(F(∇u))) - norm(∂Ψuu(F(∇u)))
  #  norm(∂Ψu(F(∇u0)))

  @test Ψ(F(∇u)) == 6440.959849358168
  @test norm(∂Ψu(F(∇u))) == 52.8548808805944
  @test isapprox(norm(∂Ψuu(F(∇u))), 18128.524371074407, rtol=1e-14)
end





@testset "Hessian∇JRegularization" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
  ∇u0 = TensorValue(1.0, 2.0, 3.0, 4.0) * 0.0
  model = ARAP2D(μ=μParams[1])
  modelreg = Hessian∇JRegularization(model)

  Ψ, ∂Ψu, ∂Ψuu = modelreg()
  F, _, J_ = get_Kinematics(modelreg.Kinematic)

  # ∂Ψu_(F) =TensorValue(ForwardDiff.gradient(x -> Ψ(x), get_array(F)))
  # ∂Ψuu_(F) =TensorValue(ForwardDiff.hessian(x -> Ψ(x), get_array(F)))
  # norm(∂Ψu_(F(∇u))) - norm(∂Ψu(F(∇u)))
  #  norm(∂Ψuu_(F(∇u))) - norm(∂Ψuu(F(∇u)))
  #  norm(∂Ψu(F(∇u0)))


  @test Ψ(F(∇u), J_(F(∇u))) == 6457.022976353012
  @test norm(∂Ψu(F(∇u), J_(F(∇u)))) == 52.980951554554586
  @test norm(∂Ψuu(F(∇u), J_(F(∇u)))) == 18172.854611409108
end



