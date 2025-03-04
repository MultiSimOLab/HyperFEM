using Mimosa
using Gridap.TensorValues



 


struct ThermoMech_EntropicPolyconvex2 
  Model1::Float64
  Model2::Float64
  β::Float64
  G::Function
  ϕ::Function
  s::Function

  function ThermoMech_EntropicPolyconvex2(; Thermo::Float64, Mechano::Float64, β::Float64, G::Function, ϕ::Function, s::Function)  
    new(Thermo, Mechano, β, G, ϕ, s)
  end

end


ThermoMech_EntropicPolyconvex2(Model1=1.0, Model2=1.0, β=1.0, G=(x)->x, ϕ=(x)->x, s=(x)->x) 








  modmec  = MoneyRivlin3D(λ=10.0, μ1=1.0, μ2=1.0, ρ=1.0)
  modterm = ThermalModel(Cv=3.4, θr=2.2, α=1.2, κ=1.0)
  β=0.7
  G(x)    =  x*(log(x) - 1.0) - 4/3*x^(3/2) + 2*x +  1/3
  γ₁      =  0.5
  γ₂      =  0.5
  γ₃      =  0.5
  s(I1,I2,I3) =  1/3*((I1/3.0)^γ₁ + (I2/3.0)^γ₂ + I3^γ₃)
  ϕ(x)        =  2.0*(x+1.0)*log(x+1.0) - 2.0*x*(1+log(2)) + 2.0*(1 - log(2))
  consmodel = ThermoMech_EntropicPolyconvex2(modterm, modmec,   β, G, ϕ, s )

  Ψ, ∂Ψu, ∂Ψθ, ∂Ψuu, ∂Ψθθ, ∂Ψuθ = consmodel(DerivativeStrategy{:autodiff}())
  Ψm, _,_=modmec(DerivativeStrategy{:analytic}())
  gradu           =  1e-1*TensorValue(1,2,3,4,5,6,7,8,9)
  δθ              =  21.6

  Ψm(gradu)
  Ψ(gradu,δθ)
  norm(∂Ψu(gradu,δθ))
  norm(∂Ψθ(gradu,δθ))
  norm(∂Ψuu(gradu,δθ))
  norm(∂Ψθθ(gradu,δθ))
  norm(∂Ψuθ(gradu,δθ))



