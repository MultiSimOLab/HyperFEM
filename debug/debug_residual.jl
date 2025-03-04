include("../src/Mimosa.jl")
using Gridap
using TimerOutputs
using LinearAlgebra
using Test
using BenchmarkTools
 



 modelMR = Mimosa.MoneyRivlin3D(3.0, 1.0, 2.0,1.0)
 modelID = Mimosa.IdealDielectric(4.0)
 modelelectro = Mimosa.ElectroMech(modelMR,modelID)

 Ψm, ∂Ψmu, ∂Ψmuu = modelMR(Mimosa.DerivativeStrategy{:analytic}())
 Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = modelelectro(Mimosa.DerivativeStrategy{:analytic}())

 partition = (100,100,100)
 pmin = Point(0.0, 0.0, 0.0)
 L = VectorValue(1, 1, 1)
 pmax = pmin + L
 model = CartesianDiscreteModel(pmin, pmax, partition)
 labels = get_face_labeling(model)
 add_tag_from_tags!(labels,"support0",[1,2,3,4,5,6,7,8])


#Define reference FE (Q2/P1(disc) pair)
order = 1
reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
reffeφ = ReferenceFE(lagrangian, Float64, order)

#Define test FESpaces
Vu = TestFESpace(model, reffeu, dirichlet_tags= Int[],   conformity=:H1)
Vφ = TestFESpace(model, reffeφ, labels=labels,   conformity=:H1)
V = MultiFieldFESpace([Vu, Vφ])
U=V
#Setup integration
degree = 2 * order
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ, 2)

  









 
function resh(uh,φh)
  res((v,vφ))= Mimosa.WeakForms.residual_EM(Mimosa.WeakForms.CouplingStrategy{:monolithic}(), (uh,φh), (v,vφ), (∂Ψu, ∂Ψφ), dΩ)
end
function resh_mech(uh,φh)
  res(v)= Mimosa.WeakForms.residual_EM(Mimosa.WeakForms.CouplingStrategy{:staggered_M}(), (uh,φh), v, ∂Ψu, dΩ)
end
function resh_electro(uh,φh)
  res(vφ)= Mimosa.WeakForms.residual_EM(Mimosa.WeakForms.CouplingStrategy{:staggered_E}(), (uh,φh), vφ, ∂Ψφ, dΩ)
end 
 
 uh = FEFunction(Vu, zeros(Float64, num_free_dofs(Vu)))
 φh = FEFunction(Vφ, zeros(Float64, num_free_dofs(Vφ)))
 uh2 = FEFunction(Vu, zeros(Float64, num_free_dofs(Vu)))

 @allocations puh=get_free_dof_values(uh)
 @allocations puh2=get_free_dof_values(uh2)
  
  puh2.+=10.0

  @allocations puh.+=puh2

  @allocations viewuh=view(uh.free_values,:)
  @allocations viewuh2=view(uh2.free_values,:)
  @allocations viewuh.+=viewuh2


  @allocations viewuh=view(uh.free_values,:)
  @allocations viewuh[1:3].=3.0

 
get_free_dof_values(uh)+= 

xx_view=view(xx, 1:3)
xx_view[1]=55.0



res_=assemble_vector(resh(uh,φh), V)
res2_=assemble_vector(resh_mech(uh,φh), Vu)
res3_=assemble_vector(resh_electro(uh,φh), Vφ)


function jac(uh,φh)
  jac((du,dφ),(v,vφ))= Mimosa.WeakForms.jacobian_EM(Mimosa.WeakForms.CouplingStrategy{:monolithic}(), (uh, φh), (du, dφ), (v, vφ), (∂Ψuu, ∂Ψφu, ∂Ψφφ), dΩ)
end 
function jac_mech(uh,φh)
  jac(du,v)= Mimosa.WeakForms.jacobian_EM(Mimosa.WeakForms.CouplingStrategy{:staggered_M}(), (uh, φh), du, v, ∂Ψuu, dΩ)
end 
function jac_electro(uh,φh)
  jac(dφ,vφ)= Mimosa.WeakForms.jacobian_EM(Mimosa.WeakForms.CouplingStrategy{:staggered_E}(), (uh, φh), dφ, vφ, ∂Ψφφ, dΩ)
end 



function jacmm(uh)
  jac(du,v)= Mimosa.WeakForms.jacobian_M(uh, du, v, ∂Ψmuu, dΩ)
end 

reset_timer!()
  jac2_=assemble_matrix(jacmm(uh), a ,Vu,Vu)
  print_timer()

norm(jac2_)

reset_timer!()
  jac_=assemble_matrix(jac_mech(uh,φh), Vu,Vu)
  print_timer()

  reset_timer!()
  jac_=assemble_matrix(jac_electro(uh,φh), Vφ,Vφ)
  print_timer()

 reset_timer!()
  jac_=assemble_matrix(jac(uh,φh), V,V)
  norm(jac_)
  print_timer()


#    modelMR = Mimosa.MoneyRivlin3D(3.0, 1.0, 2.0)
#    modelID = Mimosa.IdealDielectric(4.0)
#    modelelectro = Mimosa.ElectroMech(modelMR,modelID)

#    Ψ, ∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = modelelectro(Mimosa.DerivativeStrategy{:analytic}())
#    Ψm, ∂Ψmu, ∂Ψmuu = modelMR(Mimosa.DerivativeStrategy{:analytic}())
#  ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3
#  ∇φ = VectorValue(1.0, 2.0, 3.0)

#  Ψ(∇u,∇φ)
#  ∂Ψu(∇u,∇φ)
#  ∂Ψφ(∇u,∇φ)
#  norm(∂Ψuu(∇u,∇φ))
#  ∂Ψφu(∇u,∇φ)
#  ∂Ψφφ(∇u,∇φ)


#  res = @benchmark Ψ(∇u,∇φ)
#  res = @benchmark ∂Ψu(∇u,∇φ)
#  res = @benchmark ∂Ψφ(∇u,∇φ)
#  res = @benchmark ∂Ψuu(∇u,∇φ)
#  res = @benchmark ∂Ψφu(∇u,∇φ)
#  res = @benchmark ∂Ψφφ(∇u,∇φ)

#  @code_warntype Ψ(∇u,∇φ)
#  @code_warntype ∂Ψu(∇u,∇φ)
#  @code_warntype ∂Ψφ(∇u,∇φ)
#  @code_warntype ∂Ψuu(∇u,∇φ)
#  @code_warntype ∂Ψφu(∇u,∇φ)
#  @code_warntype ∂Ψφφ(∇u,∇φ)
 
# # norm(∂Ψuu_mr(∇u))
# reset_timer!()
# for i in 1:1e6
#   Ψ(∇u,∇φ)
#   ∂Ψu(∇u,∇φ)
#   ∂Ψφ(∇u,∇φ)
#   ∂Ψuu(∇u,∇φ)
#   ∂Ψφu(∇u,∇φ)
#   ∂Ψφφ(∇u,∇φ)
#  end
# print_timer()
 
# reset_timer!()
# for i in 1:1e6
#   Ψm(∇u)
#   ∂Ψmu(∇u)
#   ∂Ψmuu(∇u)
#  end
# print_timer()

 
# using Gridap
# using LinearAlgebra: tr
# using Mimosa
# using TimerOutputs
# using SparseArrays

# modelMR = Mimosa.MoneyRivlin3D(3.0, 1.0, 2.0)
# Ψ_mr, ∂Ψu_mr, ∂Ψuu_mr= modelMR(Mimosa.DerivativeStrategy{:analytic}())
# L = VectorValue(1, 1, 1)
# n=80
# partition = (n, n, n)
# pmin = Point(0.0, 0.0, 0.0)
# pmax = pmin + L
# model = CartesianDiscreteModel(pmin, pmax, partition )
# labeling = get_face_labeling(model)
# add_tag_from_tags!(labeling, "support0", [1, 2, 3, 4, 5, 6, 7, 8])

# order = 1
# reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
# V = TestFESpace(model, reffeu, labels=labeling, dirichlet_tags=["support0"], conformity=:H1)
# u0 = VectorValue(0.0, 0.0, 0.0)
# U = TrialFESpace(V, [u0])

# degree = 2 * order
# Ωₕ = Triangulation(model)
# dΩ = Measure(Ωₕ, degree)


# x0 = zeros(Float64, num_free_dofs(V))
# ndofm::Int = num_free_dofs(V)
# uh = FEFunction(U, x0[1:ndofm])

# v= get_fe_basis(V)
# du = get_trial_fe_basis(V)
# a=  ∫(∇(v)'  ⊙ ((∂Ψuu_mr ∘ (∇(uh)')) ⊙ (∇(du)'))) * dΩ

# jac(du, v)=  ∫(∇(v)'  ⊙ ((∂Ψuu_mr ∘ (∇(uh)')) ⊙ (∇(du)'))) * dΩ

# # cell_mat = a[Ωₕ]
# # cell_dofs= get_cell_dof_ids(V)

# # assem=SparseMatrixAssembler(V,V)
# # data = ([cell_mat],[cell_dofs],[cell_dofs])
# # reset_timer!()
# # A= assemble_matrix(assem,data)
# # print_timer()

# function run(jac, U, V)
#    A=assemble_matrix(jac, U, V)
# end

# function runrun()
# reset_timer!()
#  run(jac, U, V)
# print_timer()
# end



 






