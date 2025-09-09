
using ..TensorAlgebra


# =====================
# Visco elastic models
# =====================

struct ViscousIncompressible{T} <: Visco
  ShortTerm::Elasto
  τ::Float64
  Kinematic::T
  function ViscousIncompressible(shortTerm, τ::Float64; Kinematic::KinematicModel=Kinematics(Visco))
    new{typeof(Kinematic)}(shortTerm, τ, Kinematic)
  end
  function (obj::ViscousIncompressible)(Λ::Float64=1.0; Δt::Float64)
    Ψe, Se, ∂Se∂Ce       = obj.ShortTerm(KinematicDescription{:SecondPiola}())
    ∂Ψ∂F(F, Fn, state)   = Piola(obj, Δt, Se, ∂Se∂Ce, F, Fn, state)
    ∂Ψ∂F∂F(F, Fn, state) = Tangent(obj, Δt, Se, ∂Se∂Ce, F, Fn, state)
    return Ψe, ∂Ψ∂F, ∂Ψ∂F∂F
  end
end

function initializeStateVariables(::ViscousIncompressible, points::Measure)
  v = VectorValue(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0)
  CellState(v, points)
end

function updateStateVariables!(obj::ViscousIncompressible, Δt, u, un, stateVar)
  F, _, _ = get_Kinematics(obj.Kinematic)
  _, Se, ∂Se∂Ce = obj.ShortTerm(KinematicDescription{:SecondPiola}())
  return_mapping(A, F, Fn) = ReturnMapping(obj, Δt, Se, ∂Se∂Ce, F, Fn, A)
  update_state!(return_mapping, stateVar, F∘∇(u)', F∘∇(un)')
end

struct GeneralizedMaxwell{T} <: ViscoElastic
  LongTerm::Elasto
  Branches::NTuple{N,Visco} where N
  Kinematic::T
  function GeneralizedMaxwell(longTerm::Elasto,branches::Visco...; Kinematic::KinematicModel=Kinematics(Elasto))
    new{typeof(Kinematic)}(longTerm,branches)
  end
  function (obj::GeneralizedMaxwell)(Λ::Float64=1.0; Δt::Float64)
    Ψe, ∂Ψeu, ∂Ψeuu = obj.LongTerm(Λ)
    DΨv = map(b -> b(Λ, Δt=Δt), obj.Branches)
    Ψα, ∂Ψαu, ∂Ψαuu = map(i -> getindex.(DΨv, i), 1:3)
    Ψ(∇u, ∇un, states...) = mapreduce((Ψi, state) -> Ψi(∇u, ∇un, state), +, Ψα, states; init=Ψe(∇u))
    ∂Ψu(∇u, ∇un, states...) = mapreduce((∂Ψiu, state) -> ∂Ψiu(∇u, ∇un, state), +, ∂Ψαu, states; init=∂Ψeu(∇u))
    ∂Ψuu(∇u, ∇un, states...) = mapreduce((∂Ψiuu, state) -> ∂Ψiuu(∇u, ∇un, state), +, ∂Ψαuu, states; init=∂Ψeuu(∇u))
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end

function initializeStateVariables(model::GeneralizedMaxwell, points::Measure)
  map(b -> initializeStateVariables(b, points), model.Branches)
end

function updateStateVariables!(model::GeneralizedMaxwell, Δt, u, un, stateVars)
  @assert length(model.Branches) == length(stateVars)
  for (branch, state) in zip(model.Branches, stateVars)
    updateStateVariables!(branch, Δt, u, un, state)
  end
end


# """
#   _getKinematic(::Visco, ∇u, Uv⁻¹)

# Compute the kinematics of a viscous model.

# # Arguments
# - `::Visco`: A viscous model
# - `∇u`: The deformation gradient at the considered time step
# - `Uv⁻¹`: The inverse of the viscous strain at the considered time step

# # Returns
# - `F`
# - `C`
# - `Ce`
# """
# function _getKinematic(::Visco, ∇u, Uv⁻¹)
#   F = one(∇u) + ∇u
#   C = F' * F
#   Ce = Uv⁻¹ * C * Uv⁻¹
#   return (F, C, Ce)
# end


"""
  ViscousStrain(Ce::TensorValue, C::TensorValue)::TensorValue
  
Get viscous Uv and its inverse.

# Arguments
- `Ce`
- `C`

# Return
- `Ue`
- `Uv`
- `invUv`
"""
function ViscousStrain(Ce, C)
  Ue = sqrt(Ce)
  Ue_C_Ue = Ue * C * Ue
  invUe = inv(Ue)
  Uv = invUe * sqrt(Ue_C_Ue) * invUe
  invUv = inv(Uv)
  return Ue, Uv, invUv
end


"""
  return_mapping_algorithm!

Compute the elastic Cauchy deformation tensor and the incompressibility condition.

# Arguments
- `obj::ViscousIncompressible`: The viscous model
- `Δt::Float64`: Time step
- `Se_::Function`: Elastic 2nd Piola-Kirchhoff stress (function of C)    
- `∂Se_∂Ce_::Function`: Derivatives of elastic 2nd Piola-Kirchhoff stress (function of C)  
- `F`: Deformation gradient
- `Ce_trial`: Elastic right Green-Cauchy at intermediate statep
- `Ce`: Elastic right Green-Cauchy deformation tensor
- `λα`: incompressibility constraint (Lagrange multiplier)

# Return
- `Ce`
- `λα`
"""
function return_mapping_algorithm!(obj::ViscousIncompressible, Δt::Float64,
                            Se::Function, ∂Se∂Ce::Function,
                            F, Ce_trial, Ce, λα)
  γα = obj.τ / (obj.τ + Δt)
  Se_trial = Se(Ce_trial)
  res, ∂res = JacobianReturnMapping(γα, Ce, Se(Ce), Se_trial, ∂Se∂Ce(Ce), F, λα)
  maxiter = 20
  tol = 1e-6
  for _ in 1:maxiter
    #---------- Update -----------#
    Δu = -∂res \ res[:]
    # Ce += reshape(Δu[1:end-1], 3, 3)
    Ce += TensorValue{3,3}(Tuple(Δu[1:end-1]))  # TODO: Check reconstruction of TensorValue. ERROR: MethodError: no method matching (TensorValue{3, 3})(::Vector{Float64})
    λα += Δu[end]
    #---- Residual and jacobian ---------#
    res, ∂res = JacobianReturnMapping(γα, Ce, Se(Ce), Se_trial, ∂Se∂Ce(Ce), F, λα)
    #---- Monitor convergence ---------#
    if norm(res) < tol
      break
    end
  end
  return Ce, λα
end


"""
Residual of the return mapping algorithm and 
its Jacobian with respect to {Ce,λα} for 
incompressible case

# Arguments

# Return
- `res`
- `∂res`
"""
function JacobianReturnMapping(γα, Ce, Se, Se_trial, ∂Se∂Ce, F, λα)
    detCe = det(Ce)
    Ge = cof(Ce)
    #--------------------------------
    # Residual
    #--------------------------------
    res1 = Se - γα * Se_trial - (1-γα) * λα * Ge
    res2 = detCe - (det(F))^2
    #--------------------------------
    # Derivatives of residual
    #--------------------------------
    ∂res1_∂Ce = ∂Se∂Ce - (1-γα) * λα * ×ᵢ⁴(Ce)
    ∂res1_∂λα = -(1-γα) * Ge
    ∂res2_∂Ce = Ge
    res = [get_array(res1)[:]; res2]  #TODO: Check the TensorValue interface
    ∂res = MMatrix{10,10}(zeros(10, 10))
    ∂res[1:9, 1:9] = get_array(∂res1_∂Ce)
    ∂res[1:9, 10] = get_array(∂res1_∂λα)[:]
    ∂res[10, 1:9] = (get_array(∂res2_∂Ce)[:])'
    return res, ∂res
end


"""
  ViscousPiola(Se::Function, Ce::SMatrix, invUv::SMatrix, F::SMatrix)::SMatrix

Viscous 1st Piola-Kirchhoff stress

# Arguments
- `Se` Elastic Piola (function of C)
- `Ce` Elastic right Green-Cauchy deformation tensor
- `invUv` Inverse of viscous strain
- `F` Deformation gradient

# Return
- `Pα::SMatrix`
"""
function ViscousPiola(Se::Function, Ce::TensorValue, invUv::TensorValue, F::TensorValue)
    Sα = invUv * Se(Ce) * invUv
    F * Sα
end


"""
  ∂Ce_∂C(::ViscousIncompressible, γα, ∂Se_∂Ce_, invUvn, Ce, Ce_trial, λα, F)

Tangent operator of Ce for the incompressible case

# Arguments
- `::ViscousIncompressible` The viscous model
- `γα`: Characteristic time τα / (τα + Δt)
- `∂Se∂Ce_`: Function of C
- ...

# Return
- `∂Ce∂C`
"""
function ∂Ce_∂C(::ViscousIncompressible, γα, ∂Se∂Ce_, invUvn, Ce, Ce_trial, λα, F)
    C = F' * F
    G = cof(C)
    Ge = cof(Ce)
    ∂Se∂Ce = ∂Se∂Ce_(Ce)
    ∂Se∂Ce_trial = ∂Se∂Ce_(Ce_trial)
    ∂Ce_trial_∂C = outer_13_24(invUvn, invUvn)
    #------------------------------------------
    # Derivative of return mapping with respect to Ce and λα
    #------------------------------------------   
    K11 = ∂Se∂Ce - (1-γα) * λα * ×ᵢ⁴(Ce)
    K12 = -(1-γα) * Ge
    K21 = Ge
    #------------------------------------------
    # Derivative of return mapping with respect to C
    #------------------------------------------   
    F1 = γα * ∂Se∂Ce_trial * ∂Ce_trial_∂C
    F2 = G
    #------------------------------------------
    # Derivative of {Ce,λα} with respect to C
    #------------------------------------------   
    K = MMatrix{10,10}(zeros(10, 10))
    K[1:9, 1:9] = get_array(K11)    # TODO: Check the TensorValue interface
    K[1:9, 10] = get_array(K12)[:]
    K[10, 1:9] = get_array(K21)[:]  # There is no need to transpose the vector
    F = [get_array(F1); (get_array(F2)[:])']
    ∂u∂C = K \ F
    ∂Ce∂C = ∂u∂C[1:9, 1:9]
    return TensorValue(∂Ce∂C)
end


"""
Tangent operator of Ce for at fixed Uv
"""
function ∂Ce_∂C_Uvfixed(invUv)
  return outer_13_24(invUv, invUv)
end


"""
∂Ce∂(Uv^{-1})
"""
function ∂Ce_∂invUv(C, invU)
  invU_C = invU * C
  outer_13_24(invU_C, I3) + outer_13_24(I3, invU_C)
end


"""
  ViscousTangentOperator::TensorValue

Tangent operator for the incompressible case

# Arguments
- `obj::ViscousIncompressible`
- `Δt::Float64`: Time step
- `Se_::Function`: Function of C
- `∂Se∂Ce_::Function`: Function of C
- `F::TensorValue`: Deformation tensor
- `Ce_trial`: Right Green-Cauchy deformation tensor at intermediate step
- `Ce`: Right Green-Cauchy deformation tensor at curent step
- `invUv`
- `invUvn`
- `λα`

# Return
- `Cv::TensorValue{9,9}`: A fourth-order tensor in flattened notation
"""
function ViscousTangentOperator(obj::ViscousIncompressible, Δt::Float64,
                  Se_::Function, ∂Se∂Ce_::Function,
                  F::TensorValue, Ce_trial, Ce, invUv, invUvn, λα)
  # -----------------------------------------
  # Characteristic time
  #------------------------------------------
  γα = obj.τ / (obj.τ + Δt)
  #------------------------------------------
  # Extract τv, Δt, μv
  #------------------------------------------
  C = F' * F
  DCe_DC = ∂Ce_∂C(obj, γα, ∂Se∂Ce_, invUvn, Ce, Ce_trial, λα, F)
  DCe_DC_Uvfixed = ∂Ce_∂C_Uvfixed(invUv)
  DCe_DinvUv = ∂Ce_∂invUv(C, invUv)
  DinvUv_DC = inv(DCe_DinvUv) * (DCe_DC - DCe_DC_Uvfixed)
  DCDF = outer_13_24(F', I3) + outer_14_23(I3, F')
  #------------------------------------------
  # 0.5*δC_{Uvfixed}:DSe[ΔC]
  #------------------------------------------
  C1 = 0.5 * DCe_DC_Uvfixed' * ∂Se∂Ce_(Ce) * DCe_DC
  #------------------------------------------
  # Se:0.5*(DUv^{-1}[ΔC]*δC*Uv^{-1} + Uv^{-1}*δC*DUv^{-1}[ΔC])
  #------------------------------------------
  invUv_Se = invUv * Se_(Ce)
  C2 = 0.5 * (contraction_IP_JPKL(invUv_Se, DinvUv_DC) +
              contraction_IP_PJKL(invUv_Se, DinvUv_DC))
  #------------------------------------------
  # Sv:(D(δC_{Uvfixed})[ΔC])
  #------------------------------------------
  Sv = invUv_Se * invUv
  C3 = outer_13_24(Sv, I3)
  #------------------------------------------
  # Total Contribution
  #------------------------------------------
  Cv = DCDF' * (C1 + C2) * DCDF + C3
  Cv
end


"""
  First Piola-Kirchhoff for the incompressible case

# Arguments
- `obj::ViscousIncompressible`: The visous model
- `Δt`: Current time step
- `Se_`: Elastic 2nd Piola (function of C)
- `∂Se∂Ce_`: 2nd Piola Derivatives (function of C)
- `F`: Current deformation gradient
- `Fn`: Previous deformation gradient
- `stateVars`: State variables (Uvα and λα)

# Return
- `Pα::Gridap.TensorValues.TensorValue`
"""
function Piola(obj::ViscousIncompressible, Δt::Float64,
                Se_::Function, ∂Se∂Ce_::Function,
                F::TensorValue, Fn::TensorValue, stateVars::VectorValue)
                # ∇u_::TensorValue, ∇un_::TensorValue, stateVars::VectorValue)
  state_vars = get_array(stateVars)
  Uvn = TensorValue{3,3}(Tuple(state_vars[1:9]))
  # Uvn = SMatrix{3,3}(state_vars[1:9])
  λαn = state_vars[10]
#   ∇u = get_array(∇u_)
#   ∇un = get_array(∇un_)
  # F = get_array(F_)
  # Fn = get_array(Fn_)
  #------------------------------------------
  # Get kinematics
  #------------------------------------------
  invUvn  = inv(Uvn)
  _, C_, Ce_ = get_Kinematics(obj.Kinematic)
  C = C_(F)
  Cn = C_(Fn)
  Ceᵗʳ = Ce_(C, invUvn)
  Cen  = Ce_(Cn, invUvn)
#   F, C, Ceᵗʳ = _getKinematic(obj, ∇u, invUvn)
#   _, _, Cen  = _getKinematic(obj, ∇un, invUvn)
  #------------------------------------------
  # Return mapping algorithm
  #------------------------------------------
  Ce, _ = return_mapping_algorithm!(obj, Δt, Se_, ∂Se∂Ce_, F, Ceᵗʳ, Cen, λαn)
  #------------------------------------------
  # Get invUv and Sα
  #------------------------------------------
  _, _, invUv = ViscousStrain(Ce, C)
  Pα = ViscousPiola(Se_, Ce, invUv, F)
  #------------------------------------------
  # Tangent operator
  #------------------------------------------
  # check_type = Pα isa TensorValue{3,3,Float64}
  # if !check_type throw("Pα is a $(typeof(Pα))") end
  Pα
  # return TensorValue(Pα)
end


"""
Visco-Elastic model for incompressible case

# Arguments
- `obj::ViscousIncompressible`: The visous model
- `Δt`: Current time step
- `Se_`: Elastic 2nd Piola (function of C)
- `∂Se∂Ce_`: 2nd Piola Derivatives (function of C)
- `∇u_`: Current deformation gradient
- `∇un_`: Previous deformation gradient
- `stateVars`: State variables (Uvα and λα)

# Return
- `Cα::Gridap.TensorValues.TensorValue`
"""
function Tangent(obj::ViscousIncompressible, Δt::Float64,
                 Se_::Function, ∂Se∂Ce_::Function,
                 F::TensorValue, Fn::TensorValue, stateVars::VectorValue)
                #  ∇u_::TensorValue, ∇un_::TensorValue, stateVars::VectorValue)
  state_vars = get_array(stateVars)
  Uvn = TensorValue{3,3}(Tuple(state_vars[1:9]))
  # Uvn = SMatrix{3,3}(state_vars[1:9])
  λαn = state_vars[10]
#   ∇u = get_array(∇u_)
#   ∇un = get_array(∇un_)
#   F = get_array(F_)
#   Fn = get_array(Fn_)
  #------------------------------------------
  # Get kinematics
  #------------------------------------------
  invUvn  = inv(Uvn)
  _, C_, Ce_ = get_Kinematics(obj.Kinematic)
  C = C_(F)
  Cn = C_(Fn)
  Ceᵗʳ = Ce_(C, invUvn)
  Cen  = Ce_(Cn, invUvn)
#   F, C, Ceᵗʳ = _getKinematic(obj, ∇u, invUvn)
#   _, _, Cen  = _getKinematic(obj, ∇un, invUvn)
  #------------------------------------------
  # Return mapping algorithm
  #------------------------------------------
  Ce, λα = return_mapping_algorithm!(obj, Δt, Se_, ∂Se∂Ce_, F, Ceᵗʳ, Cen, λαn)
  #------------------------------------------
  # Get invUv and Sα
  #------------------------------------------
  _, _, invUv = ViscousStrain(Ce, C)
  #------------------------------------------
  # Tangent operator
  #------------------------------------------
  Cα = ViscousTangentOperator(obj, Δt, Se_, ∂Se∂Ce_, F, Ceᵗʳ, Ce, invUv, invUvn, λα)
  return Cα
end


"""
    Return mapping for the incompressible case

    # Arguments
    - `::ViscousIncompressible`
    - `Δt::Float64`: Time step
    - `Se_::Function`: Elastic Piola (function of C)
    - `∂Se∂Ce_::Function`: Piola Derivatives (function of C)
    - `∇u_::TensorValue`
    - `∇un_::TensorValue`
    - `stateVars::VectorValue`: State variables (10-component vector gathering Uvα and λα)

    # Return
    - `::bool`: indicates whether the state variables should be updated
    - `::VectorValue`: State variables at new time
"""
function ReturnMapping(obj::ViscousIncompressible, Δt::Float64,
                       Se_::Function, ∂Se∂Ce_::Function,
                       F::TensorValue, Fn::TensorValue, stateVars::VectorValue)
  state_vars = get_array(stateVars)
  Uvn = TensorValue{3,3}(Tuple(state_vars[1:9]))
  # Uvn = SMatrix{3,3}(state_vars[1:9])
  λαn = state_vars[10]
  # ∇u = get_array(∇u_)
  # ∇un = get_array(∇un_)
  # F = get_array(F_)
  # Fn = get_array(Fn_)
  #------------------------------------------
  # Get kinematics
  #------------------------------------------
  invUvn  = inv(Uvn)
  _, C_, Ce_ = get_Kinematics(obj.Kinematic)
  C = C_(F)
  Cn = C_(Fn)
  Ceᵗʳ = Ce_(C, invUvn)
  Cen  = Ce_(Cn, invUvn)
#   F, C, Ceᵗʳ = _getKinematic(obj, ∇u, invUvn)
#   _, _, Cen  = _getKinematic(obj, ∇un, invUvn)
  #------------------------------------------
  # Return mapping algorithm
  #------------------------------------------
  Ce, λα = return_mapping_algorithm!(obj, Δt, Se_, ∂Se∂Ce_, F, Ceᵗʳ, Cen, λαn)
  #------------------------------------------
  # Get Uv and λα
  #------------------------------------------
  _, Uv, _ = ViscousStrain(Ce, C)
  Cell_ = [get_array(Uv)[:]; λα]  # TODO: Another problem with TensorValue slice
  return true, VectorValue(Cell_)
end


function ViscousDissipation(obj::ViscousIncompressible, Δt::Float64,
                       Se_::Function, ∂Se∂Ce_::Function,
                       F::TensorValue, Fn::TensorValue, stateVars::VectorValue)
  state_vars = get_array(stateVars)
  Uvn = TensorValue{3,3}(Tuple(state_vars[1:9]))
  # Uvn = SMatrix{3,3}(state_vars[1:9])
  λαn = state_vars[10]
  # ∇u = get_array(∇u_)
  # ∇un = get_array(∇un_)
  # F = get_array(F_)
  # Fn = get_array(Fn_)
  #------------------------------------------
  # Get kinematics
  #------------------------------------------
  invUvn  = inv(Uvn)
  _, C_, Ce_ = get_Kinematics(obj.Kinematic)
  C = C_(F)
  Cn = C_(Fn)
  Ceᵗʳ = Ce_(C, invUvn)
  Cen  = Ce_(Cn, invUvn)
#   F, C, Ceᵗʳ = _getKinematic(obj, ∇u, invUvn)
#   _, _, Cen  = _getKinematic(obj, ∇un, invUvn)
  #------------------------------------------
  # Return mapping algorithm
  #------------------------------------------
  Ce, λα = return_mapping_algorithm!(obj, Δt, Se_, ∂Se∂Ce_, F, Ceᵗʳ, Cen, λαn)
  τ = obj.τ
  Se = Se_(Ce)
  invCCe = inv(2*∂Se∂Ce_(Ce))
  ∂Se = -1/τ * (Se - λα*cof(Ce))
  Dvis = -Se ⊙ (invCCe ⊙ ∂Se)
  Dvis
end

