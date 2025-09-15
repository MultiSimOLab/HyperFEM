using Gridap.TensorValues
using HyperFEM.TensorAlgebra
using Test
using BenchmarkTools

 

@testset "Jacobian regularization" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  F = one(∇u) + ∇u
  J = det(F)
  @test J == 1.0149819999999996
  @test logreg(J; Threshold=0.01) == 0.014870878346353422
end


@testset "outer" begin
  A = TensorValue(1.0, 2.0, 3.0, 4.0)
  B = TensorValue(5.0, 6.0, 7.0, 8.0)
  u = VectorValue(1.0, 2.0)
  v = VectorValue(3.0, 4.0)
  @test u ⊗ v   == TensorValue(3.0, 6.0, 4.0, 8.0)
  @test u ⊗₁² v == TensorValue(3.0, 6.0, 4.0, 8.0)
  @test A ⊗ B     == TensorValue(5.0, 10.0, 15.0, 20.0, 6.0, 12.0, 18.0, 24.0, 7.0, 14.0, 21.0, 28.0, 8.0, 16.0, 24.0, 32.0)
  @test A ⊗₁₂³⁴ B == TensorValue(5.0, 10.0, 15.0, 20.0, 6.0, 12.0, 18.0, 24.0, 7.0, 14.0, 21.0, 28.0, 8.0, 16.0, 24.0, 32.0)
  @test A ⊗₁₃²⁴ B == TensorValue(5.0, 10.0, 6.0, 12.0, 15.0, 20.0, 18.0, 24.0, 7.0, 14.0, 8.0, 16.0, 21.0, 28.0, 24.0, 32.0)
  @test A ⊗₁₄²³ B == TensorValue(5.0, 10.0, 6.0, 12.0, 7.0, 14.0, 8.0, 16.0, 15.0, 20.0, 18.0, 24.0, 21.0, 28.0, 24.0, 32.0)
  @test u ⊗₁²³ A == TensorValue{2,4}(1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0)
  @test A ⊗₁₂³ u == TensorValue{2,4}(1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0)
  @test A ⊗₁₃² u == TensorValue{2,4}(1.0, 2.0, 2.0, 4.0, 3.0, 4.0, 6.0, 8.0)
end


# @benchmark (A ⊗₁₃²⁴ B)
# @benchmark (A ⊗₁₂³ V1)
# @benchmark (A ⊗₁₃² V1)
# @benchmark (V1 ⊗₁²³ A)
# @benchmark (A ⊗₁₃²⁴ B)
# @benchmark (D × A)

# @code_warntype (A ⊗₁₃²⁴ B)
# @code_warntype (A ⊗₁₂³ V1)
# @code_warntype (A ⊗₁₃² V1)
# @code_warntype (V1 ⊗₁²³ A)
# @code_warntype (A ⊗₁₃²⁴ B)
# @code_warntype (D × A)

 

@testset "cross" begin
  A = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  B = TensorValue(4.6, 2.1, 1.7, 3.2, 6.5, 1.4, 9.2, 8.0, 9.0) * 1e-3
  C = A ⊗ B
  D = TensorValue([4.6 2.1 1.7 3.2 6.5 1.4 9.2 8.0 9.0;
  1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0;
  5.3 2.0 3.1 1.9 5.4 9.8 0.4 8.8 3.1] * 1e-3)
  @test norm(×ᵢ⁴(A)) == 0.033763886032268264
  @test norm(A × B) == 6.246230863488799e-5
  @test norm(C × B) == 2.4491455542698976e-6
  @test norm(B × C) == 1.104276381618298e-6
  @test norm(D × A) == 0.00012378691368638284
  @test norm(get_array(A) × get_array(B))== 6.246230863488799e-5


  

 end

@testset "inner" begin
  A = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  B = TensorValue(4.6, 2.1, 1.7, 3.2, 6.5, 1.4, 9.2, 8.0, 9.0) * 1e-3
  C = A ⊗ B
  D = TensorValue([4.6 2.1 1.7 3.2 6.5 1.4 9.2 8.0 9.0;
    1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0;
    5.3 2.0 3.1 1.9 5.4 9.8 0.4 8.8 3.1] * 1e-3)
  E = VectorValue(1.0, 2.0, 3.0) * 1e-3
  @test norm(C ⊙ A) == 4.676298215469156e-6
  @test norm(D ⊙ E) == 0.00010313946868197451
  @test norm(D ⊙ A) == 0.0004509607632599537
end

@testset "sum" begin
  A = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  B = TensorValue(4.6, 2.1, 1.7, 3.2, 6.5, 1.4, 9.2, 8.0, 9.0) * 1e-3
  @test norm(A + B) == 0.03393449572337859
end


@testset "Identity" begin
  I2_ = TensorValue(1.0, 0.0, 0.0, 1.0)
  I3_ = TensorValue(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
  I4_ = TensorValue(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
  I9_ = TensorValue(
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
  @test I2_ == I2
  @test I3_ == I3
  @test I4_ == I4
  @test I9_ == I9
  # @benchmark I2_
  # @benchmark I2
  # @benchmark I9_
  # @benchmark I9
end
