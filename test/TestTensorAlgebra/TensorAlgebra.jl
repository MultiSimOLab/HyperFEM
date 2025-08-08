using Gridap.TensorValues

 

@testset "Jacobian regularization" begin
  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  F = one(∇u) + ∇u
  J = det(F)
  @test J == 1.0149819999999996
  @test logreg(J; Threshold=0.01) == 0.014870878346353422
end

@testset "outer" begin
  A = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
  B = TensorValue(4.6, 2.1, 1.7, 3.2, 6.5, 1.4, 9.2, 8.0, 9.0) * 1e-3
  V1 = VectorValue(1.0, 2.0, 3.0)
  V2 = VectorValue(1.5, 2.5, 3.5)
  @test norm(A ⊗ B) == 0.0002984572833756952
  @test norm(A ⊗₁₂³⁴ B)== 0.0002984572833756952
  @test norm(V1 ⊗ V2) == 17.04406054905931
  @test norm(V1 ⊗₁² V2) == 17.04406054905931
  @test norm(V1 ⊗₁²³ A)== 0.06316644678941503
  @test norm(A ⊗₁₂³ V1)== 0.06316644678941503
  @test norm(A ⊗₁₃² V1)== 0.06316644678941503
  @test norm(A ⊗₁₃²⁴ B)== 0.00029845728337569516
  @test norm(get_array(V1) ⊗ get_array(V2)) == 17.04406054905931

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

