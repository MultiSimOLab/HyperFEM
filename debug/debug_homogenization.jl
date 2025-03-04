using Gridap, GridapGmsh, GridapMultiSimO, GridapSolvers, DrWatson, TimerOutputs
using GridapSolvers.NonlinearSolvers
using GridapSolvers.LinearSolvers

using Gridap.FESpaces
using GridapMultiSimO: jacobian, IterativeSolver, solve!, update_state!
using WriteVTK
using Revise


 ∇u = TensorValue(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) * 1e-3

Fmacro = TensorValue(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


Kmodel=EvolutiveKinematics(Mechano)

 


 
 
physmodel = MoneyRivlin3D(λ=3.0, μ1=1.0, μ2=0.0, Kinematic= Kmodel)




_,∂Ψu,_= physmodel(1.0)
∂Ψu(∇u)
 
 

a=TensorValue{3, 3, Float64, 9}(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    (Ψ(∇u))
   Fmacro*=2.0
   norm(∂Ψu(∇u))
  norm(∂Ψuu(∇u))
 

h(Λ)=((F)->F*Λ)

  f(u)=2.0*u


 
f(3.0) |> g

gg(u)=(g1 ∘ f)(u)

gg(3.0) 



 
