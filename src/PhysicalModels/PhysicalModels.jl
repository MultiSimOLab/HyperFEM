module PhysicalModels

using Gridap
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

export PhysicalModel
export Mechano
export Elasto
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

export KinematicDescription
export DerivativeStrategy

export initializeStateVariables
export updateStateVariables!
export update_state!

export Kinematics
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
abstract type Visco <: Mechano end
abstract type ViscoElastic <: Mechano end

abstract type MultiPhysicalModel <: PhysicalModel end
abstract type ElectroMechano <: MultiPhysicalModel end
abstract type ThermoElectroMechano <: MultiPhysicalModel end
abstract type ThermoMechano <: MultiPhysicalModel end
abstract type ThermoElectro <: MultiPhysicalModel end
abstract type FlexoElectro <: MultiPhysicalModel end
abstract type MagnetoMechano <: MultiPhysicalModel end

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
# State variables management
# ============================================

function initializeStateVariables(::PhysicalModel, points::Measure)
  return nothing
end

function updateStateVariables!(::Any, ::PhysicalModel, vars...)
end

end