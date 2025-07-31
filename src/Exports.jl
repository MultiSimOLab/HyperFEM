
macro publish(mod, name)
  quote
    using HyperFEM.$mod: $name
    export $name
  end
end

@publish TensorAlgebra (*)
@publish TensorAlgebra (×ᵢ⁴)
@publish TensorAlgebra (⊗₁₂³)
@publish TensorAlgebra (⊗₁₃²)
@publish TensorAlgebra (⊗₁²³)
@publish TensorAlgebra (⊗₁₃²⁴)
@publish TensorAlgebra (⊗₁₂³⁴)
@publish TensorAlgebra (⊗₁²)
@publish TensorAlgebra logreg
@publish TensorAlgebra Box
@publish TensorAlgebra Ellipsoid
@publish TensorAlgebra I9
@publish TensorAlgebra Tensorize


@publish PhysicalModels DerivativeStrategy
@publish PhysicalModels LinearElasticity3D
@publish PhysicalModels LinearElasticity2D
@publish PhysicalModels NeoHookean3D
@publish PhysicalModels IncompressibleNeoHookean3D
@publish PhysicalModels IncompressibleNeoHookean2D
@publish PhysicalModels IncompressibleNeoHookean2D_CV
@publish PhysicalModels IncompressibleNeoHookean3D_2dP
@publish PhysicalModels MoneyRivlin3D
@publish PhysicalModels MoneyRivlin2D
@publish PhysicalModels NonlinearMoneyRivlin3D
@publish PhysicalModels NonlinearMoneyRivlin2D
@publish PhysicalModels NonlinearMoneyRivlin2D_CV
@publish PhysicalModels NonlinearNeoHookean_CV
@publish PhysicalModels NonlinearIncompressibleMoneyRivlin2D_CV
@publish PhysicalModels TransverseIsotropy3D
@publish PhysicalModels ThermalModel
@publish PhysicalModels IdealDielectric
@publish PhysicalModels IdealMagnetic
@publish PhysicalModels IdealMagnetic2D
@publish PhysicalModels HardMagnetic
@publish PhysicalModels HardMagnetic2D
@publish PhysicalModels ElectroMechModel
@publish PhysicalModels ThermoElectroMechModel
@publish PhysicalModels ThermoMechModel
@publish PhysicalModels ThermoMech_EntropicPolyconvex
@publish PhysicalModels FlexoElectroModel
@publish PhysicalModels ThermoElectroMech_Govindjee
@publish PhysicalModels ThermoElectroMech_PINNs
@publish PhysicalModels MagnetoMechModel
@publish PhysicalModels MagnetoVacuumModel
@publish PhysicalModels ARAP2D
@publish PhysicalModels ARAP2D_regularized
@publish PhysicalModels HessianRegularization
@publish PhysicalModels Hessian∇JRegularization

@publish PhysicalModels Mechano
@publish PhysicalModels Thermo
@publish PhysicalModels Electro
@publish PhysicalModels ThermoMechano
@publish PhysicalModels ElectroMechano
@publish PhysicalModels MagnetoMechano
@publish PhysicalModels ThermoElectro
@publish PhysicalModels FlexoElectro
@publish PhysicalModels ThermoElectroMechano
@publish PhysicalModels EnergyInterpolationScheme
@publish PhysicalModels update_state!
@publish PhysicalModels Kinematics
@publish PhysicalModels EvolutiveKinematics
@publish PhysicalModels get_Kinematics
@publish PhysicalModels getIsoInvariants
 


@publish WeakForms residual
@publish WeakForms jacobian
@publish WeakForms mass_term

@publish ComputationalModels  DirichletBC
@publish ComputationalModels  NeumannBC
@publish ComputationalModels  get_Neumann_dΓ
@publish ComputationalModels  residual_Neumann
@publish ComputationalModels  NothingBC
@publish ComputationalModels  MultiFieldBC
@publish ComputationalModels  SingleFieldTC
@publish ComputationalModels  MultiFieldTC
@publish ComputationalModels  TrialFESpace
@publish ComputationalModels  get_state
@publish ComputationalModels  get_measure
@publish ComputationalModels  get_spaces
@publish ComputationalModels  get_trial_space
@publish ComputationalModels  get_test_space
@publish ComputationalModels  get_assemblers
@publish ComputationalModels  StaticNonlinearModel
@publish ComputationalModels  DynamicNonlinearModel
@publish ComputationalModels  StaticLinearModel
@publish ComputationalModels  solve!
@publish ComputationalModels  dirichlet_preconditioning!
@publish ComputationalModels  GmshDiscreteModel
@publish ComputationalModels  updateBC!
@publish ComputationalModels  PostProcessor
@publish ComputationalModels  StaggeredModel
@publish ComputationalModels  Cauchy
@publish ComputationalModels  Entropy
@publish ComputationalModels  D0
@publish ComputationalModels  reset!
@publish ComputationalModels  DirichletCoupling
@publish ComputationalModels  evaluate!
@publish ComputationalModels  InterpolableBC
@publish ComputationalModels  InterpolableBC!

@publish Solvers IterativeSolver
@publish Solvers Newton_RaphsonSolver
@publish Solvers Injectivity_Preserving_LS
@publish Solvers Roman_LS
@publish Solvers update_cellstate!

 


# @publish LinearSolvers solve
# @publish LinearSolvers solve!