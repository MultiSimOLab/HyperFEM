
# ===================
# Electrical models
# ===================

struct IdealDielectric{A} <: Electro
  ε::Float64
  Kinematic::A
  function IdealDielectric(; ε::Float64, Kinematic::KinematicModel=Kinematics(Electro))
    new{typeof(Kinematic)}(ε, Kinematic)
  end
end
