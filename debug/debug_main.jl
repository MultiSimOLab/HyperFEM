# # include("../src/Mimosa.jl")
# using Mimosa

# main(;problemName="TEMStaticSquare", 
#       ptype="ThermoElectroMechanics",
#       couplingstrategy="monolithic")
  
 


struct MoneyRivlin3D
    λ::Float64
    μ1::Float64
    μ2::Float64
end


model1=MoneyRivlin3D(1.0,2.0,3.0)
model2=MoneyRivlin3D(1.0,1.0,3.5)
