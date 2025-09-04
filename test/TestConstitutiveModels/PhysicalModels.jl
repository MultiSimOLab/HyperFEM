using Gridap





@testset "NonlinearMooneyRivlin_CV" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3

  model = NonlinearMooneyRivlin_CV(λ=3.0, μ1=1.0, μ2=1.0, α=2.0, β=1.0, γ=6.0)

  Ψ, ∂Ψu, ∂Ψuu = model()
  F, _, _ = get_Kinematics(model.Kinematic)

  # @benchmark norm(∂Ψuu(F(∇u)))
  # @code_warntype  norm(∂Ψuu(F(∇u)))

  #  ∂Ψu_(F) =TensorValue(ForwardDiff.gradient(x -> Ψ(x), get_array(F)))
  #  ∂Ψuu_(F) =TensorValue(ForwardDiff.hessian(x -> Ψ(x), get_array(F)))

  #  norm(∂Ψu_(F(∇u))) -norm(∂Ψu(F(∇u)))
  #  norm(∂Ψuu_(F(∇u))) -norm(∂Ψuu(F(∇u)))



  @test Ψ(F(∇u)) == 8.274742322531269
  @test norm(∂Ψu(F(∇u))) == 5.647570016731348
  @test norm(∂Ψuu(F(∇u))) == 653.1484437383998

end



@testset "NonlinearNeoHookean_CV" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3

  model = NonlinearNeoHookean_CV(λ=3.0, μ=1.0, α=2.0, γ=6.0)

  Ψ, ∂Ψu, ∂Ψuu = model()
  F, _, _ = get_Kinematics(model.Kinematic)

  #  ∂Ψu_(F) =TensorValue(ForwardDiff.gradient(x -> Ψ(x), get_array(F)))
  #  ∂Ψuu_(F) =TensorValue(ForwardDiff.hessian(x -> Ψ(x), get_array(F)))

  #  norm(∂Ψu_(F(∇u))) -norm(∂Ψu(F(∇u)))
  #  norm(∂Ψuu_(F(∇u))) -norm(∂Ψuu(F(∇u)))

  @test Ψ(F(∇u)) == 6.774247349061977
  @test norm(∂Ψu(F(∇u))) == 5.578324662235092
  @test norm(∂Ψuu(F(∇u))) == 645.1103360183206

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
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
  model = LinearElasticity2D(λ=3.0, μ=1.0)
  Ψ, ∂Ψu, ∂Ψuu = model()
  F, _, _ = get_Kinematics(model.Kinematic)
  @test (Ψ(F(∇u))) == 6.699999999999821e-5
  @test norm(∂Ψu(F(∇u))) == 0.029461839725311915
  @test norm(∂Ψuu(F(∇u))) == 8.48528137423857
end

@testset "LinearElasticity3D" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  model = LinearElasticity3D(λ=3.0, μ=1.0)
  Ψ, ∂Ψu, ∂Ψuu = model()
  F, _, _ = get_Kinematics(model.Kinematic)

  @test (Ψ(F(∇u))) == 0.0006104999999999824
  @test norm(∂Ψu(F(∇u))) == 0.09933277404764056
  @test norm(∂Ψuu(F(∇u))) == 11.874342087037917
end


@testset "NeoHookean3D" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  model = NeoHookean3D(λ=3.0, μ=1.0)
  Ψ, ∂Ψu, ∂Ψuu = model()
  F, _, _ = get_Kinematics(model.Kinematic)
  @test Ψ(F(∇u)) == 0.0006083121396460722
  @test norm(∂Ψu(F(∇u))) == 0.099612127449168118
  @test norm(∂Ψuu(F(∇u))) == 12.073268944343628
end


@testset "MooneyRivlin2D" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
  model = MooneyRivlin2D(λ=3.0, μ1=1.0, μ2=2.0)
  Ψ, ∂Ψu, ∂Ψuu = model()
  F, _, _ = get_Kinematics(model.Kinematic)
  @test Ψ(F(∇u)) == 4.000175692713462
  @test norm(∂Ψu(F(∇u))) == 0.07481942119475013
  @test norm(∂Ψuu(F(∇u))) == 21.74472726344396
end

@testset "MooneyRivlin3D" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  model = MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0)
  Ψ, ∂Ψu, ∂Ψuu = model()
  F, _, _ = get_Kinematics(model.Kinematic)
  @test Ψ(F(∇u)) == 0.001598259078230413
  @test norm(∂Ψu(F(∇u))) == 0.24833325775972206
  @test norm(∂Ψuu(F(∇u))) == 30.36786840739546
end


@testset "NonlinearMooneyRivlin2D" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
  ∇u0 = TensorValue(0.0, 0.0, 0.0, 0.0) * 1e-3

  μParams = [6456.9137547089595, 896.4633794151492,
    1.999999451256222,
    1.9999960497608036,
    11747.646562400318,
    0.7841068624959612, 1.5386288924587603]
  model = NonlinearMooneyRivlin2D(λ=(μParams[1] + μParams[2]) * 1e2, μ1=μParams[1], μ2=μParams[2], α=μParams[3], β=μParams[4])

  Ψ, ∂Ψu, ∂Ψuu = model()
  F, _, _ = get_Kinematics(model.Kinematic)

  # ∂Ψu_(F) =TensorValue(ForwardDiff.gradient(x -> Ψ(x), get_array(F)))
  # ∂Ψuu_(F) =TensorValue(ForwardDiff.hessian(x -> Ψ(x), get_array(F)))

  # norm(∂Ψu_(F(∇u))) -norm(∂Ψu(F(∇u)))
  # norm(∂Ψuu_(F(∇u))) -norm(∂Ψuu(F(∇u)))

  @test Ψ(F(∇u)) == 5524.542944928555
  @test norm(∂Ψu(F(∇u))) == 5322.887691298287
  @test norm(∂Ψuu(F(∇u))) == 1.5136029879040532e6


end




@testset "Yeoh3D" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3

  model = Yeoh3D(λ=3.0, C10=1.0, C20=1.0, C30=1.0)
  Ψ, ∂Ψu, ∂Ψuu = model()

  F, _, _ = get_Kinematics(model.Kinematic)

  #    ∂Ψu_(F) =TensorValue(ForwardDiff.gradient(x -> Ψ(x), get_array(F)))
  #  ∂Ψuu_(F) =TensorValue(ForwardDiff.hessian(x -> Ψ(x), get_array(F)))

  #  norm(∂Ψu_(F(∇u))) -norm(∂Ψu(F(∇u)))
  #  norm(∂Ψuu_(F(∇u))) -norm(∂Ψuu(F(∇u)))



  @test Ψ(F(∇u)) == 0.0018248918516909718
  @test norm(∂Ψu(F(∇u))) == 0.33825920882848515
  @test norm(∂Ψuu(F(∇u))) == 40.845986774511886
end




@testset "NonlinearMooneyRivlin2D_CV" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
  ∇u0 = TensorValue(0.0, 0.0, 0.0, 0.0) * 1e-3

  μParams = [6456.9137547089595, 896.4633794151492,
    1.999999451256222,
    1.9999960497608036,
    11747.646562400318,
    0.7841068624959612, 1.5386288924587603]
  model = NonlinearMooneyRivlin2D_CV(λ=(μParams[1] + μParams[2]) * 1e2, μ1=μParams[1], μ2=μParams[2], α=μParams[3], β=μParams[4], γ=μParams[4])

  Ψ, ∂Ψu, ∂Ψuu = model()
  F, _, _ = get_Kinematics(model.Kinematic)

  # ∂Ψu_(F) =TensorValue(ForwardDiff.gradient(x -> Ψ(x), get_array(F)))
  # ∂Ψuu_(F) =TensorValue(ForwardDiff.hessian(x -> Ψ(x), get_array(F)))

  # norm(∂Ψu_(F(∇u))) -norm(∂Ψu(F(∇u)))
  # norm(∂Ψuu_(F(∇u))) -norm(∂Ψuu(F(∇u)))

  @test Ψ(F(∇u)) == 1.47626389512027e6
  @test norm(∂Ψu(F(∇u))) == 41486.412892304914
  @test norm(∂Ψuu(F(∇u))) == 1.171028839080193e7


end



@testset "NonlinearMooneyRivlin3D" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3

  μParams = [6456.9137547089595, 896.4633794151492,
    1.999999451256222,
    1.9999960497608036,
    11747.646562400318,
    0.7841068624959612, 1.5386288924587603]
  model = NonlinearMooneyRivlin3D(λ=(μParams[1] + μParams[2]) * 1e2, μ1=μParams[1], μ2=μParams[2], α=μParams[3], β=μParams[4])

  Ψ, ∂Ψu, ∂Ψuu = model()
  F, _, _ = get_Kinematics(model.Kinematic)

  # ∂Ψu_(F) =TensorValue(ForwardDiff.gradient(x -> Ψ(x), get_array(F)))
  # ∂Ψuu_(F) =TensorValue(ForwardDiff.hessian(x -> Ψ(x), get_array(F)))

  # norm(∂Ψu_(F(∇u))) -norm(∂Ψu(F(∇u)))
  # norm(∂Ψuu_(F(∇u))) -norm(∂Ψuu(F(∇u)))

  @test Ψ(F(∇u)) == 5600.526853280392
  @test norm(∂Ψu(F(∇u))) == 19622.49002309361
  @test norm(∂Ψuu(F(∇u))) == 2.313386876448522e6

end


@testset "IncompressibleNeoHookean2D" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
  ∇u0 = TensorValue(1.0, 2.0, 3.0, 4.0) * 0.0

  μParams = [6456.9137547089595, 896.4633794151492,
    1.999999451256222,
    1.9999960497608036,
    11747.646562400318,
    0.7841068624959612, 1.5386288924587603]
  model = IncompressibleNeoHookean2D(λ=(μParams[1] + μParams[2]) * 1e2, μ=μParams[1])

  Ψ, ∂Ψu, ∂Ψuu = model()
  F, _, J_ = get_Kinematics(model.Kinematic)

  # ∂Ψu_(F) =TensorValue(ForwardDiff.gradient(x -> Ψ(x), get_array(F)))
  # ∂Ψuu_(F) =TensorValue(ForwardDiff.hessian(x -> Ψ(x), get_array(F)))

  # norm(∂Ψu_(F(∇u))) - norm(∂Ψu(F(∇u)))
  #  norm(∂Ψuu_(F(∇u))) - norm(∂Ψuu(F(∇u)))

  @test Ψ(F(∇u)) == 9678.62138906247
  @test norm(∂Ψu(F(∇u))) == 5225.228283839419
  @test norm(∂Ψuu(F(∇u))) == 1.4859216839324834e6
end

@testset "IncompressibleNeoHookean2D_CV" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
  ∇u0 = TensorValue(1.0, 2.0, 3.0, 4.0) * 0.0

  μParams = [6456.9137547089595, 896.4633794151492,
    1.999999451256222,
    1.9999960497608036,
    11747.646562400318,
    0.7841068624959612, 1.5386288924587603]
  model = IncompressibleNeoHookean2D_CV(λ=(μParams[1] + μParams[2]) * 1e2, μ=μParams[1], γ=3.0)

  Ψ, ∂Ψu, ∂Ψuu = model()
  F, _, J_ = get_Kinematics(model.Kinematic)

  ∂Ψu_(F) = TensorValue(ForwardDiff.gradient(x -> Ψ(x), get_array(F)))
  ∂Ψuu_(F) = TensorValue(ForwardDiff.hessian(x -> Ψ(x), get_array(F)))

  norm(∂Ψu_(F(∇u))) - norm(∂Ψu(F(∇u)))
  norm(∂Ψuu_(F(∇u))) - norm(∂Ψuu(F(∇u)))

  @test Ψ(F(∇u)) == 1.4805254328174442e6
  @test norm(∂Ψu(F(∇u))) == 93109.55194565043
  @test norm(∂Ψuu(F(∇u))) == 2.6282687148741454e7
end


@testset "NonlinearIncompressibleMooneyRivlin2D_CV" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
  ∇u0 = TensorValue(1.0, 2.0, 3.0, 4.0) * 0.0

  μParams = [6456.9137547089595, 896.4633794151492,
    1.999999451256222,
    1.9999960497608036,
    11747.646562400318,
    0.7841068624959612, 1.5386288924587603]
  model = NonlinearIncompressibleMooneyRivlin2D_CV(λ=(μParams[1] + μParams[2]) * 1e2, μ=μParams[1], α=μParams[3], γ=3.0)

  Ψ, ∂Ψu, ∂Ψuu = model()
  F, _, J_ = get_Kinematics(model.Kinematic)

  # ∂Ψu_(F) =TensorValue(ForwardDiff.gradient(x -> Ψ(x), get_array(F)))
  # ∂Ψuu_(F) =TensorValue(ForwardDiff.hessian(x -> Ψ(x), get_array(F)))

  # norm(∂Ψu_(F(∇u))) - norm(∂Ψu(F(∇u)))
  #  norm(∂Ψuu_(F(∇u))) - norm(∂Ψuu(F(∇u)))


  @test Ψ(F(∇u)) == 1.4853683856379557e6
  @test norm(∂Ψu(F(∇u))) == 93139.40884969762
  @test norm(∂Ψuu(F(∇u))) == 2.629114137261709e7
end





@testset "TransverseIsotropy2D" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
  ∇u0 = TensorValue(1.0, 2.0, 3.0, 4.0) * 0.0

  N = VectorValue(1.0, 2.0) / sqrt(5.0)
  μParams = [6456.9137547089595, 896.4633794151492,
    1.999999451256222,
    1.9999960497608036,
    11747.646562400318,
    0.7841068624959612, 1.5386288924587603]
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
  μParams = [6456.9137547089595, 896.4633794151492,
    1.999999451256222,
    1.9999960497608036,
    11747.646562400318,
    0.7841068624959612, 1.5386288924587603]
  model = TransverseIsotropy3D(μ=μParams[5], α=μParams[6], β=μParams[7])

  Ψ, ∂Ψu, ∂Ψuu = model()
  F, _, _ = get_Kinematics(model.Kinematic)
  @test Ψ(F(∇u), N) == 269927.3350807581
  @test norm(∂Ψu(F(∇u), N)) == 947447.8711645481
  @test norm(∂Ψuu(F(∇u), N)) == 3.8258646319087776e6
end

@testset "ElectroMechano" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  ∇φ = VectorValue(1.0, 2.0, 3.0)
  modelMR = MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0)
  modelID = IdealDielectric(ε=4.0)
  modelelectro = ElectroMechModel(Mechano=modelMR, Electro=modelID)
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


@testset "TermoElectroMech" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  ∇φ = VectorValue(1.0, 2.0, 3.0)
  θt = 3.4 - 1.0
  modelMR = MooneyRivlin3D(λ=3.0, μ1=1.0, μ2=2.0)
  modelID = IdealDielectric(ε=4.0)
  modelT = ThermalModel(Cv=1.0, θr=1.0, α=2.0)
  f(δθ::Float64)::Float64 = (δθ + 1.0) / 1.0
  df(δθ::Float64)::Float64 = 1.0
  modelTEM = ThermoElectroMechModel(Thermo=modelT, Electro=modelID, Mechano=modelMR, fθ=f, dfdθ=df)
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
  modelTM = ThermoMechModel(Thermo=modelT, Mechano=modelMR, fθ=f, dfdθ=df)
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
  consmodel = ThermoMech_EntropicPolyconvex(Thermo=modterm, Mechano=modmec, β=β, G=G, ϕ=ϕ, s=s)

  Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ = consmodel()
  F, _, _ = get_Kinematics(modmec.Kinematic)

  @test (Ψ(F(∇u), θt)) == -129.4022076861008
  @test norm(∂Ψu(F(∇u), θt)) == 437.9269386687991
  @test norm(∂Ψθ(F(∇u), θt)) == 13.97666807099424
  @test norm(∂Ψuu(F(∇u), θt)) == 2066.7910102392775
  @test norm(∂Ψθθ(F(∇u), θt)) == 0.46689338540182707
  @test norm(∂Ψuθ(F(∇u), θt)) == 14.243050132210923

end


@testset "FlexoElectroMechanics" begin

  # Constitutive models
  ∇umacro = TensorValue(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) * 1e-2
  ∇φ = VectorValue(1.0, 2.0, 3.0)
  ∇u = 1e-1 * TensorValue(1, 2, 3, 4, 5, 6, 7, 8, 9)
  Emacro = VectorValue(0.0, 0.0, sqrt((1.0 + 5.0) / (1.0 + 5.0)) * 0.1)
  A = TensorValue{3,9,Float64,27}(0.0013981268088158305, 0.0008195783555664171,
    0.0016562357569609649, 0.0008406006468943406, 0.0009224862278332126, 0.001155322042969417,
    0.0005129360612093835, 0.0012909164959851265, 0.001152698427032676, 0.0008406006468943406,
    0.0009224862278332126, 0.001155322042969417, 0.00034502469077903774, 0.00021859521770246592,
    0.0017683239822952042, 0.0009471782270005929, 0.001800950730156155, 0.0009587801251013468,
    0.0005129360612093835, 0.0012909164959851265, 0.001152698427032676, 0.0009471782270005929,
    0.001800950730156155, 0.0009587801251013468, 0.0008421896546088605, 0.0007114140805416631,
    0.001245006227831607)
  Kin_mec = EvolutiveKinematics(Mechano; F=(t) -> ((∇u, x) -> ∇u + one(∇u) + t * ∇umacro + t * (A ⊙ x)))
  Kin_elec = EvolutiveKinematics(Electro; E=(t) -> ((∇φ) -> -∇φ + t * Emacro))

  physmec = MooneyRivlin3D(λ=10.0, μ1=1.0, μ2=1.0, Kinematic=Kin_mec)
  physelec = IdealDielectric(ε=1.0, Kinematic=Kin_elec)
  physmodel = FlexoElectroModel(Mechano=physmec, Electro=physelec, κ=1000.0)

  Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ, Φ = physmodel(1.0)

  F, _, _ = get_Kinematics(physmec.Kinematic; Λ=1.0)
  E = get_Kinematics(physelec.Kinematic; Λ=1.0)
  X = VectorValue(2.4, 1.9, 3.3)

  @test (Ψ(F(∇u, X), E(∇φ))) == 13.408299698687056
  @test norm(∂Ψu(F(∇u, X), E(∇φ))) == 58.375248703633474
  @test norm(∂Ψφ(F(∇u, X), E(∇φ))) == 1.2365693126167825
  @test norm(∂Ψuu(F(∇u, X), E(∇φ))) == 208.40589433833898
  @test norm(∂Ψφφ(F(∇u, X), E(∇φ))) == 3.8963298254031042
  @test norm(∂Ψφu(F(∇u, X), E(∇φ))) == 5.910650247536949

end



@testset "ThermoElectroMech_Bonet" begin

  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  ∇φ = VectorValue(1.0, 2.0, 3.0)
  θt = 3.4 - 1.0
  modelMR = MooneyRivlin3D(λ=0.0, μ1=0.5, μ2=0.5)
  modelID = IdealDielectric(ε=1.0)
  modelT = ThermalModel(Cv=17.385, θr=293.0, α=0.00156331, γv=2.0, γd=2.0)

  modelTEM = ThermoElectroMech_Bonet(Thermo=modelT, Electro=modelID, Mechano=modelMR)
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

  modelTEM = ThermoElectroMech_Govindjee(Thermo=modelT, Electro=modelID, Mechano=modelMR, fθ=f, dfdθ=df, gθ=g, dgdθ=dg, β=0.0)
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

  data_filename = "test/models/test_NN_TEM.json"
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

  @test Ψ(F(∇u), H0(∇φ)) == -3.123376791098092e-6
  @test norm(∂Ψu(F(∇u), H0(∇φ))) == 4.406161902404201e-6
  @test norm(∂Ψφ(F(∇u), H0(∇φ))) == 2.793633631007779e-6
  @test norm(∂Ψuu(F(∇u), H0(∇φ))) == 1.0766129497574707e-5
  @test norm(∂Ψφu(F(∇u), H0(∇φ))) == 5.589596497314291e-6
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
  modelmagneto = MagnetoMechModel(Mechano=modelMR, Magneto=modelID)
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







@testset "IdealMagnetic_SoftMaterial2D" begin

  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
  ∇φ = VectorValue(1.0, 2.0)
  N = VectorValue(0.0, 1.0)

  modelMR = MooneyRivlin2D(λ=3.0, μ1=1.0, μ2=2.0)
  modelID = IdealMagnetic2D(μ=1.2566e-6, χe=0.0)
  modelmagneto = MagnetoMechModel(Mechano=modelMR, Magneto=modelID)
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
  modelmagneto = MagnetoMechModel(Mechano=modelMR, Magneto=modelID)
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
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
  ∇u0 = TensorValue(1.0, 2.0, 3.0, 4.0) * 0.0

  μParams = [6456.9137547089595, 896.4633794151492,
    1.999999451256222,
    1.9999960497608036,
    11747.646562400318,
    0.7841068624959612, 1.5386288924587603]
  model = ARAP2D(μ=μParams[1])

  Ψ, ∂Ψu, ∂Ψuu = model()
  F, _, J_ = get_Kinematics(model.Kinematic)

  # ∂Ψu_(F) =TensorValue(ForwardDiff.gradient(x -> Ψ(x), get_array(F)))
  # ∂Ψuu_(F) =TensorValue(ForwardDiff.hessian(x -> Ψ(x), get_array(F)))

  # norm(∂Ψu_(F(∇u))) - norm(∂Ψu(F(∇u)))
  #  norm(∂Ψuu_(F(∇u))) - norm(∂Ψuu(F(∇u)))

  @test Ψ(F(∇u)) == 6457.022976353012
  @test norm(∂Ψu(F(∇u))) == 52.980951554554586
  @test norm(∂Ψuu(F(∇u))) == 18172.854404408303
end




@testset "ARAP2D_regularized" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
  ∇u0 = TensorValue(1.0, 2.0, 3.0, 4.0) * 0.0

  μParams = [6456.9137547089595, 896.4633794151492,
    1.999999451256222,
    1.9999960497608036,
    11747.646562400318,
    0.7841068624959612, 1.5386288924587603]
  model = ARAP2D_regularized(μ=μParams[1])

  Ψ, ∂Ψu, ∂Ψuu = model()
  F, _, J_ = get_Kinematics(model.Kinematic)

  # ∂Ψu_(F) =TensorValue(ForwardDiff.gradient(x -> Ψ(x), get_array(F)))
  # ∂Ψuu_(F) =TensorValue(ForwardDiff.hessian(x -> Ψ(x), get_array(F)))

  # norm(∂Ψu_(F(∇u))) - norm(∂Ψu(F(∇u)))
  #  norm(∂Ψuu_(F(∇u))) - norm(∂Ψuu(F(∇u)))
  #  norm(∂Ψu(F(∇u0)))


  @test Ψ(F(∇u)) == 6440.959849358168
  @test norm(∂Ψu(F(∇u))) == 52.8548808805944
  @test norm(∂Ψuu(F(∇u))) == 18128.952058660318
end


@testset "HessianRegularization" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
  ∇u0 = TensorValue(1.0, 2.0, 3.0, 4.0) * 0.0

  μParams = [6456.9137547089595, 896.4633794151492,
    1.999999451256222,
    1.9999960497608036,
    11747.646562400318,
    0.7841068624959612, 1.5386288924587603]
  model = ARAP2D_regularized(μ=μParams[1])
  modelreg = HessianRegularization(Mechano=model)

  Ψ, ∂Ψu, ∂Ψuu = modelreg()
  F, _, J_ = get_Kinematics(modelreg.Kinematic)

  # ∂Ψu_(F) =TensorValue(ForwardDiff.gradient(x -> Ψ(x), get_array(F)))
  # ∂Ψuu_(F) =TensorValue(ForwardDiff.hessian(x -> Ψ(x), get_array(F)))
  # norm(∂Ψu_(F(∇u))) - norm(∂Ψu(F(∇u)))
  #  norm(∂Ψuu_(F(∇u))) - norm(∂Ψuu(F(∇u)))
  #  norm(∂Ψu(F(∇u0)))

  @test Ψ(F(∇u)) == 6440.959849358168
  @test norm(∂Ψu(F(∇u))) == 52.8548808805944
  @test norm(∂Ψuu(F(∇u))) == 18128.524371074407
end





@testset "Hessian∇JRegularization" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0) * 1e-3
  ∇u0 = TensorValue(1.0, 2.0, 3.0, 4.0) * 0.0

  μParams = [6456.9137547089595, 896.4633794151492,
    1.999999451256222,
    1.9999960497608036,
    11747.646562400318,
    0.7841068624959612, 1.5386288924587603]
  model = ARAP2D(μ=μParams[1])
  modelreg = Hessian∇JRegularization(Mechano=model)

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



