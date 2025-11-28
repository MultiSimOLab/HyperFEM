module PhysicalModels

using Gridap
using Gridap.CellData
using Gridap.Helpers

using DrWatson
using ForwardDiff
using LinearAlgebra
using ..TensorAlgebra
using ..TensorAlgebra: _∂H∂F_2D
using ..TensorAlgebra: trAA
using StaticArrays

import Base: +

export Yeoh3D
export NeoHookean3D
export IncompressibleNeoHookean3D
export IncompressibleNeoHookean2D
export IncompressibleNeoHookean2D_CV
export IncompressibleNeoHookean3D_2dP
export ARAP2D
export ARAP2D_regularized
export VolumetricEnergy
export MooneyRivlin3D
export MooneyRivlin2D
export NonlinearMooneyRivlin3D
export NonlinearMooneyRivlin2D
export NonlinearMooneyRivlin2D_CV
export NonlinearNeoHookean_CV
export NonlinearMooneyRivlin_CV
export NonlinearIncompressibleMooneyRivlin2D_CV
export EightChain
export TransverseIsotropy3D
export TransverseIsotropy2D
export LinearElasticity3D
export LinearElasticity2D
export Magnetic
export IdealDielectric
export IdealMagnetic
export IdealMagnetic2D
export HardMagnetic
export HardMagnetic2D
export ThermalModel
export ElectroMechModel
export ThermoElectroMechModel
export ThermoMechModel
export ThermoMech_EntropicPolyconvex
export FlexoElectroModel
export ThermoElectroMech_Govindjee
export ThermoElectroMech_PINNs
export ThermoElectroMech_Bonet
export MagnetoMechModel
export GeneralizedMaxwell
export ViscousIncompressible
export HGO_4Fibers
export HGO_1Fiber

export PhysicalModel
export Mechano
export Elasto
export AnisoElastic
export Visco
export ViscoElastic
export Electro
export Magneto
export Thermo
export ElectroMechano
export MagnetoMechano
export ThermoElectroMechano
export ThermoMechano
export ThermoElectro
export FlexoElectro
export EnergyInterpolationScheme
export SecondPiola

export DerivativeStrategy

export initializeStateVariables
export updateStateVariables!
export update_state!
export update_time_step!

export Kinematics
export KinematicDescription
export Solid
export KinematicModel
export EvolutiveKinematics
export get_Kinematics
export getIsoInvariants

export HessianRegularization
export Hessian∇JRegularization

struct DerivativeStrategy{Kind} end

abstract type PhysicalModel end
abstract type Mechano <: PhysicalModel end
abstract type Electro <: PhysicalModel end
abstract type Magneto <: PhysicalModel end
abstract type Thermo <: PhysicalModel end

abstract type Elasto <: Mechano end
abstract type IsoElastic <: Elasto end
abstract type AnisoElastic <: Elasto end
abstract type Visco <: Mechano end
abstract type ViscoElastic <: Mechano end

abstract type MultiPhysicalModel <: PhysicalModel end
abstract type ElectroMechano <: MultiPhysicalModel end
abstract type ThermoElectroMechano <: MultiPhysicalModel end
abstract type ThermoMechano <: MultiPhysicalModel end
abstract type ThermoElectro <: MultiPhysicalModel end
abstract type FlexoElectro <: MultiPhysicalModel end
abstract type MagnetoMechano <: MultiPhysicalModel end
abstract type InternalFibers end

include("KinematicModels.jl")

include("MechanicalModels.jl")

include("ViscousModels.jl")

include("MagneticModels.jl")

include("ElectricalModels.jl")

include("ThermalModels.jl")

include("ThermoMechanicalModels.jl")

include("ElectroMechanicalModels.jl")

include("MagnetoMechanicalModels.jl")

include("ThermoElectroMechanicalModels.jl")

include("PINNs.jl")


# ============================================
# Physical models interface
# ============================================

"""
Initialize the state variables for the given constitutive model and discretization.
"""
function initializeStateVariables(::PhysicalModel, points::Measure)
  return nothing
end


"""
Update the state variables. The state variables must be initialized using the function 'initializeStateVariables'.
"""
function updateStateVariables!(::Any, ::PhysicalModel, vars...)
end


"""
Return the dissipation and its derivatives if any.
"""
function Dissipation(::PhysicalModel, args...)
  D(::Any...) = 0.0
end


"""
Return the energy density and its derivatives as functions of C instead of F.
"""
function SecondPiola(::T, args...) where {T<:PhysicalModel}
  throw("The function 'SecondPiola' has not been implemented for $T.")
end


"""
Set the time step to be used internally by the constitutive model.
"""
function update_time_step!(::PhysicalModel, Δt::Float64)
  Δt
end

end
