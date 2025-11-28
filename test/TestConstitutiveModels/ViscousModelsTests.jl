
using Gridap.TensorValues
using Gridap.Arrays
using HyperFEM.TensorAlgebra
using HyperFEM.PhysicalModels
using StaticArrays
using Test


μ      = 1.367e4  # Pa
N      = 7.860e5  # -
λ      = μ*100    # Pa
μ1     = 3.153e5  # Pa
τ1     = 10.72    # s
μ2     = 5.639e5  # Pa
τ2     = 0.82     # s
μ3     = 1.981e5  # Pa
τ3     = 498.8    # s


function isochoric_F(F)
  J = det(F)
  @assert J > 0 "Non-physical deformation, got det(F) < 0 (det(F) = $J)"
  J^(-1/3) * F
end


function numerical_piola(Ψ, F, ϵ=1e-6)
  P = MMatrix{3,3}(zeros(Float64,9))
  for i in 1:9
    Fp = mutable(F)
    Fm = mutable(F)
    Fp[i] += ϵ
    Fm[i] -= ϵ
    Ψp = Ψ(TensorValue(Fp))
    Ψm = Ψ(TensorValue(Fm))
    P[i] = (Ψp - Ψm) / 2ϵ
  end
  return TensorValue(P)
end


function numerical_tangent(Ψ, F, ϵ=1e-5)
  H = MMatrix{9,9}(zeros(Float64,81))
  for j in 1:9
    ej = zeros(9); ej[j] = ϵ
    for i in 1:9
      ei = zeros(9); ei[i] = ϵ
      if i == j
        Ψp = Ψ(F + TensorValue(ei...))
        Ψ0 = Ψ(F)
        Ψm = Ψ(F - TensorValue(ei...))
        H[i,j] = (Ψp - 2Ψ0 + Ψm) / ϵ^2
      else
        Ψpp = Ψ(F + TensorValue(( ei+ej)...))
        Ψpm = Ψ(F + TensorValue(( ei-ej)...))
        Ψmp = Ψ(F + TensorValue((-ei+ej)...))
        Ψmm = Ψ(F + TensorValue((-ei-ej)...))
        H[i,j] = (Ψpp - Ψpm - Ψmp + Ψmm) / 4ϵ^2
      end
    end
  end
  return TensorValue(H)
end


function richardson_expansion(func, x, ϵ)
  (4.0func(x,ϵ) - func(x,2ϵ)) / 3.0
end


function test_viscous_derivatives_numerical(model; rtolP=1e-12, rtolH=1e-12)
  update_time_step!(model, 1e-2)
  Ψ, ∂Ψu, ∂Ψuu = model()
  F = TensorValue(1.:9...) * 1e-3 + I3
  Fn = TensorValue(1.:9...) * 5e-4 + I3
  Uvn = isochoric_F(TensorValue(1.,2.,3.,2.,4.,5.,3.,5.,6.) * 2e-4 + I3)
  λvn = 1e-3
  Avn = VectorValue(Uvn..., λvn)
  piola = richardson_expansion((F, ϵ) -> numerical_piola(F -> Ψ(F, Fn, Avn), F, ϵ), F, 1e-5)
  tangent = richardson_expansion((F, ϵ) -> numerical_tangent(F -> Ψ(F, Fn, Avn), F, ϵ), F, 1e-4)
  @test isapprox(∂Ψu(F, Fn, Avn), piola, rtol=rtolP)
  @test isapprox(∂Ψuu(F, Fn, Avn), tangent, rtol=rtolH)
end


function test_elastic_derivatives_numerical(model; rtolP=1e-12, rtolH=1e-12)
  Ψ, ∂Ψu, ∂Ψuu = model()
  F = TensorValue(1.:9...) * 1e-3 + I3
  piola = richardson_expansion((F,ϵ) -> numerical_piola(Ψ,F,ϵ), F, 1e-5)
  tangent = richardson_expansion((F,ϵ) -> numerical_tangent(Ψ,F,ϵ), F, 1e-4)
  @test isapprox(∂Ψu(F), piola, rtol=rtolP)
  @test isapprox(∂Ψuu(F), tangent, rtol=rtolH)
end


struct EmptyElastic <: Elasto
  function (::EmptyElastic)()
    Ψ(F) = 0.0
    ∂Ψu(F) = TensorValue(zeros(9)...)
    ∂Ψuu(F) = TensorValue(zeros(81)...)
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end


@testset "VolumetricEnergy" begin
  hyper_elastic_model = VolumetricEnergy(λ=λ)
  test_elastic_derivatives_numerical(hyper_elastic_model, rtolP=1e-10, rtolH=1e-9)
end;

@testset "EightChain" begin
  hyper_elastic_model = EightChain(μ=μ, N=N)
  test_elastic_derivatives_numerical(hyper_elastic_model, rtolP=1e-3, rtolH=1e-2)
end;

@testset "EightChain+VolumetricEnergy" begin
  hyper_elastic_model = EightChain(μ=μ, N=N) + VolumetricEnergy(λ=λ)
  test_elastic_derivatives_numerical(hyper_elastic_model, rtolP=1e-5, rtolH=1e-4)
end;

@testset "NeoHookean3D" begin
  hyper_elastic_model = NeoHookean3D(λ=λ, μ=μ)
  test_elastic_derivatives_numerical(hyper_elastic_model, rtolP=1e-10, rtolH=1e-9)
end;

@testset "ViscousIncompressible" begin
  visco = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μ1), τ=τ1)
  test_viscous_derivatives_numerical(visco, rtolP=1e-3, rtolH=1e-3)
end

@testset "ViscousIncompressible2" begin
  visco = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=1.0), τ=10.0)
  update_time_step!(visco, 0.1)
  Ψ, ∂Ψu, ∂Ψuu = visco()
  F    =  1e-2*TensorValue(1,2,3,4,5,6,7,8,9) + I3
  Fn   =  5e-3*TensorValue(1,2,3,4,5,6,7,8,9) + I3
  Fvn  =  2e-2*TensorValue(1.0,2.0,3.0,4.0,5.0,8.7,6.5,4.3,6.5) + I3
  Cvn  =  Fvn'*Fvn
  Uvn  =  sqrt(Cvn)
  Avn  =  VectorValue(Uvn...,0.0)
  @test isapprox(norm(∂Ψu(F, Fn, Avn)), 0.20303772905627682, rtol=1e-10)
  @test isapprox(norm(∂Ψuu(F, Fn, Avn)), 4.847586088299776, rtol=1e-10)
end

@testset "GeneralizedMaxwell EightChain 0-branch" begin
  hyper_elastic_model = EightChain(μ=μ, N=N) + VolumetricEnergy(λ=λ)
  cons_model = GeneralizedMaxwell(hyper_elastic_model)
  test_viscous_derivatives_numerical(cons_model, rtolP=1e-5, rtolH=1e-4)
end;

@testset "GeneralizedMaxwell NeoHookean 0-branch" begin
  hyper_elastic_model = NeoHookean3D(λ=λ, μ=μ)
  cons_model = GeneralizedMaxwell(hyper_elastic_model)
  test_viscous_derivatives_numerical(cons_model, rtolP=1e-10, rtolH=1e-9)
end

@testset "GeneralizedMaxwell NeoHookean 1-branch" begin
  hyper_elastic_model = NeoHookean3D(λ=λ, μ=μ)
  branch1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μ1), τ=τ1)
  cons_model = GeneralizedMaxwell(hyper_elastic_model, branch1)
  test_viscous_derivatives_numerical(cons_model, rtolP=1e-3, rtolH=1e-2)
end

@testset "Dissipation ViscousIncompressible" begin
  branch1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μ1), τ=τ1)
  D = Dissipation(branch1, 0.1)
  F = I3
  Fn = I3
  A = VectorValue(I3..., 0)
  @test D(F, Fn, A) < 1e-6
end
