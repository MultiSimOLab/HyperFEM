
using ..TensorAlgebra


# =====================
# Visco elastic models
# =====================

struct ViscousIncompressible <: Visco
  elasto::Elasto
  τ::Float64
  Δt::Ref{Float64}
  function ViscousIncompressible(elasto; τ::Float64)
    new(elasto, τ, 0)
  end
  function (obj::ViscousIncompressible)(Λ::Float64=1.0; Δt::Float64)
    @warn "The argument 'Δt' will be removed shortly. Just kept to avoid breaking benchmarks..."
    obj.Δt[] = Δt
    Ψe, Se, ∂Se∂Ce   = SecondPiola(obj.elasto)
    Ψ(F, Fn, A)      = Energy(obj, Ψe, Se, ∂Se∂Ce, F, Fn, A)
    ∂Ψ∂F(F, Fn, A)   = Piola(obj, Se, ∂Se∂Ce, F, Fn, A)
    ∂Ψ∂F∂F(F, Fn, A) = Tangent(obj, Se, ∂Se∂Ce, F, Fn, A)
    return Ψ, ∂Ψ∂F, ∂Ψ∂F∂F
  end
end

function initializeStateVariables(::ViscousIncompressible, points::Measure)
  v = VectorValue(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0)
  CellState(v, points)
end

function updateStateVariables!(state, obj::ViscousIncompressible, Δt, F, Fn)
  @warn "The argument 'Δt' will be removed shortly. Just kept to avoid breaking benchmarks..."
  obj.Δt[] = Δt
  _, Se, ∂Se∂Ce = SecondPiola(obj.elasto)
  return_mapping(A, F, Fn) = ReturnMapping(obj, Se, ∂Se∂Ce, F, Fn, A)
  update_state!(return_mapping, state, F, Fn)
end

function Dissipation(obj::ViscousIncompressible, Δt)
  @warn "The argument 'Δt' will be removed shortly. Just kept to avoid breaking benchmarks..."
  obj.Δt[] = Δt
  _, Se, ∂Se∂Ce = SecondPiola(obj.elasto)
  D(F, Fn, A) = ViscousDissipation(obj, Se, ∂Se∂Ce, F, Fn, A)
end

struct GeneralizedMaxwell <: ViscoElastic
  longterm::Elasto
  branches::NTuple{N,Visco} where N
  Δt::Ref{Float64}
  function GeneralizedMaxwell(longTerm::Elasto, branches::Visco...)
    new(longTerm,branches,0)
  end
  function (obj::GeneralizedMaxwell)(Λ::Float64=1.0; Δt::Float64)
    @warn "The argument 'Δt' will be removed shortly. Just kept to avoid breaking benchmarks..."
    obj.Δt[] = Δt
    Ψe, ∂Ψeu, ∂Ψeuu = obj.longterm()
    DΨv = map(b -> b(Δt=Δt), obj.branches)
    Ψα, ∂Ψαu, ∂Ψαuu = map(i -> getindex.(DΨv, i), 1:3)
    Ψα, ∂Ψαu, ∂Ψαuu = transpose(DΨv)
    Ψ(F, Fn, A...) = mapreduce((Ψi, Ai) -> Ψi(F, Fn, Ai), +, Ψα, A; init=Ψe(F))
    ∂Ψu(F, Fn, A...) = mapreduce((∂Ψiu, Ai) -> ∂Ψiu(F, Fn, Ai), +, ∂Ψαu, A; init=∂Ψeu(F))
    ∂Ψuu(F, Fn, A...) = mapreduce((∂Ψiuu, Ai) -> ∂Ψiuu(F, Fn, Ai), +, ∂Ψαuu, A; init=∂Ψeuu(F))
    return (Ψ, ∂Ψu, ∂Ψuu)
  end
end

function set_time_step!(obj::GeneralizedMaxwell, Δt::Float64)
  obj.Δt[] = Δt
  foreach(b -> b.Δt[] = Δt, obj.branches)
  Δt
end

function initializeStateVariables(obj::GeneralizedMaxwell, points::Measure)
  map(b -> initializeStateVariables(b, points), obj.branches)
end

function updateStateVariables!(states, obj::GeneralizedMaxwell, Δt, F, Fn)
  @warn "The argument 'Δt' will be removed shortly. Just kept to avoid breaking benchmarks..."
  obj.Δt[] = Δt
  @assert length(obj.branches) == length(states)
  for (branch, state) in zip(obj.branches, states)
    updateStateVariables!(state, branch, Δt, F, Fn)
  end
end

function Dissipation(obj::GeneralizedMaxwell, Δt)
  @warn "The argument 'Δt' will be removed shortly. Just kept to avoid breaking benchmarks..."
  obj.Δt[] = Δt
  Dα = map(b -> Dissipation(b, Δt), obj.branches)
  D(F, Fn, A...) = mapreduce((Di, Ai) -> Di(F, Fn, Ai), +, Dα, A)
end


# =====================
# Internal functions
# =====================


"""Right Cauchy-Green deformation tensor."""
function Cauchy(F::TensorValue)
  F' · F
end


"""Elastic right Cauchy-Green deformation tensor."""
function ElasticCauchy(C::TensorValue, Uv⁻¹::TensorValue)
  Uv⁻¹' · C · Uv⁻¹
end

"""
Multiplicative decomposition of visous strain.

# Return
- `Ue::TensorValue`
- `Uv::TensorValue`
- `Uv⁻¹::TensorValue`
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
function return_mapping_algorithm!(obj::ViscousIncompressible,
                            Se::Function, ∂Se∂Ce::Function,
                            C, Ce_trial, Ce, λα)
  γα = obj.τ / (obj.τ + obj.Δt[])
  Se_trial = Se(Ce_trial)
  res, ∂res = JacobianReturnMapping(γα, Ce, Se(Ce), Se_trial, ∂Se∂Ce(Ce), C, λα)
  maxiter = 20
  tol = 1e-6
  for _ in 1:maxiter
    #---------- Update -----------#
    Δu = -∂res \ res[:]
    Ce += TensorValue{3,3}(Tuple(Δu[1:end-1]))  # TODO: Check reconstruction of TensorValue. ERROR: MethodError: no method matching (TensorValue{3, 3})(::Vector{Float64})
    λα += Δu[end]
    #---- Residual and jacobian ---------#
    res, ∂res = JacobianReturnMapping(γα, Ce, Se(Ce), Se_trial, ∂Se∂Ce(Ce), C, λα)
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
function JacobianReturnMapping(γα, Ce, Se, Se_trial, ∂Se∂Ce, C, λα)
  Ge = cof(Ce)
  #--------------------------------
  # Residual
  #--------------------------------
  res1 = Se - γα * Se_trial - (1-γα) * λα * Ge
  res2 = det(Ce) - det(C)
  #--------------------------------
  # Derivatives of residual
  #--------------------------------
  ∂res1_∂Ce = ∂Se∂Ce - (1-γα) * λα * ×ᵢ⁴(Ce)
  ∂res1_∂λα = -(1-γα) * Ge
  ∂res2_∂Ce = Ge
  res = [get_array(res1)[:]; res2]
  ∂res = MMatrix{10,10}(zeros(10, 10))  # TODO: It'd be nice to use hvcat: ∂res = [∂res1_Ce ∂res1_∂λα; ∂res2_∂Ce 0.0]
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
  Sα = invUv' * Se(Ce) * invUv
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
  ∂Ce_trial_∂C = invUvn ⊗₁₃²⁴ invUvn
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
Tangent operator of Ce at fixed Uv
"""
function ∂Ce_∂C_Uvfixed(invUv)
  invUv ⊗₁₃²⁴ invUv
end


"""
∂Ce∂(Uv^{-1})
"""
function ∂Ce_∂invUv(C, invU)
  invU_C = invU * C
  invU_C ⊗₁₃²⁴ I3 + I3 ⊗₁₃²⁴ invU_C
end


"""
  ViscousTangentOperator::TensorValue

Tangent operator for the incompressible case

# Arguments
- `obj::ViscousIncompressible`
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
function ViscousTangentOperator(obj::ViscousIncompressible,
                  Se_::Function, ∂Se∂Ce_::Function,
                  F::TensorValue, Ce_trial, Ce, invUv, invUvn, λα)
  # -----------------------------------------
  # Characteristic time
  #------------------------------------------
  γα = obj.τ / (obj.τ + obj.Δt[])
  #------------------------------------------
  # Elastic tensor and derivatives
  #------------------------------------------
  C = Cauchy(F)
  DCe_DC = ∂Ce_∂C(obj, γα, ∂Se∂Ce_, invUvn, Ce, Ce_trial, λα, F)
  DCe_DC_Uvfixed = ∂Ce_∂C_Uvfixed(invUv)
  DCe_DinvUv = ∂Ce_∂invUv(C, invUv)
  DinvUv_DC = inv(DCe_DinvUv) * (DCe_DC - DCe_DC_Uvfixed)
  DCDF = F' ⊗₁₃²⁴ I3 + I3 ⊗₁₄²³ F'
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
  C3 = I3 ⊗₁₃²⁴ Sv
  #------------------------------------------
  # Total Contribution
  #------------------------------------------
  Cv = DCDF' * (C1 + C2) * DCDF + C3
  Cv
end


function Energy(obj::ViscousIncompressible,
                Ψe::Function, Se_::Function, ∂Se∂Ce_::Function,
                F::TensorValue, Fn::TensorValue, A::VectorValue)
  Uvn = TensorValue{3,3}(A[1:9]...)
  λαn = A[10]
  #------------------------------------------
  # Get kinematics
  #------------------------------------------
  invUvn = inv(Uvn)
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  Ceᵗʳ = ElasticCauchy(C, invUvn)
  Cen  = ElasticCauchy(Cn, invUvn)
  #------------------------------------------
  # Return mapping algorithm
  #------------------------------------------
  Ce, _ = return_mapping_algorithm!(obj, Se_, ∂Se∂Ce_, C, Ceᵗʳ, Cen, λαn)
  #------------------------------------------
  # Elastic energy
  #------------------------------------------
  Ψe(Ce)
end


"""
  First Piola-Kirchhoff for the incompressible case

# Arguments
- `obj::ViscousIncompressible`: The visous model
- `Se_`: Elastic 2nd Piola (function of C)
- `∂Se∂Ce_`: 2nd Piola Derivatives (function of C)
- `F`: Current deformation gradient
- `Fn`: Previous deformation gradient
- `A`: State variables (Uvα and λα)

# Return
- `Pα::Gridap.TensorValues.TensorValue`
"""
function Piola(obj::ViscousIncompressible,
                Se_::Function, ∂Se∂Ce_::Function,
                F::TensorValue, Fn::TensorValue, A::VectorValue)
  Uvn = TensorValue{3,3}(A[1:9]...)
  λαn = A[10]
  #------------------------------------------
  # Get kinematics
  #------------------------------------------
  invUvn  = inv(Uvn)
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  Ceᵗʳ = ElasticCauchy(C, invUvn)
  Cen  = ElasticCauchy(Cn, invUvn)
  #------------------------------------------
  # Return mapping algorithm
  #------------------------------------------
  Ce, _ = return_mapping_algorithm!(obj, Se_, ∂Se∂Ce_, C, Ceᵗʳ, Cen, λαn)
  #------------------------------------------
  # Get invUv and Pα
  #------------------------------------------
  _, _, invUv = ViscousStrain(Ce, C)
  Pα = ViscousPiola(Se_, Ce, invUv, F)
  Pα
end


"""
Visco-Elastic model for incompressible case

# Arguments
- `obj::ViscousIncompressible`: The visous model
- `Se_`: Elastic 2nd Piola (function of C)
- `∂Se∂Ce_`: 2nd Piola Derivatives (function of C)
- `∇u_`: Current deformation gradient
- `∇un_`: Previous deformation gradient
- `A`: State variables (Uvα and λα)

# Return
- `Cα::Gridap.TensorValues.TensorValue`
"""
function Tangent(obj::ViscousIncompressible,
                 Se_::Function, ∂Se∂Ce_::Function,
                 F::TensorValue, Fn::TensorValue, A::VectorValue)
  Uvn = TensorValue{3,3}(A[1:9]...)
  λαn = A[10]
  #------------------------------------------
  # Get kinematics
  #------------------------------------------
  invUvn  = inv(Uvn)
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  Ceᵗʳ = ElasticCauchy(C, invUvn)
  Cen  = ElasticCauchy(Cn, invUvn)
  #------------------------------------------
  # Return mapping algorithm
  #------------------------------------------
  Ce, λα = return_mapping_algorithm!(obj, Se_, ∂Se∂Ce_, C, Ceᵗʳ, Cen, λαn)
  #------------------------------------------
  # Get invUv and Sα
  #------------------------------------------
  _, _, invUv = ViscousStrain(Ce, C)
  #------------------------------------------
  # Tangent operator
  #------------------------------------------
  Cα = ViscousTangentOperator(obj, Se_, ∂Se∂Ce_, F, Ceᵗʳ, Ce, invUv, invUvn, λα)
  return Cα
end


"""
    Return mapping for the incompressible case

    # Arguments
    - `::ViscousIncompressible`
    - `Se_::Function`: Elastic Piola (function of C)
    - `∂Se∂Ce_::Function`: Piola Derivatives (function of C)
    - `∇u_::TensorValue`
    - `∇un_::TensorValue`
    - `A::VectorValue`: State variables (10-component vector gathering Uvα and λα)

    # Return
    - `::bool`: indicates whether the state variables should be updated
    - `::VectorValue`: State variables at new time step
"""
function ReturnMapping(obj::ViscousIncompressible,
                       Se_::Function, ∂Se∂Ce_::Function,
                       F::TensorValue, Fn::TensorValue, A::VectorValue)
  Uvn = TensorValue{3,3}(A[1:9]...)
  λαn = A[10]
  #------------------------------------------
  # Get kinematics
  #------------------------------------------
  invUvn  = inv(Uvn)
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  Ceᵗʳ = ElasticCauchy(C, invUvn)
  Cen  = ElasticCauchy(Cn, invUvn)
  #------------------------------------------
  # Return mapping algorithm
  #------------------------------------------
  Ce, λα = return_mapping_algorithm!(obj, Se_, ∂Se∂Ce_, C, Ceᵗʳ, Cen, λαn)
  #------------------------------------------
  # Get Uv and λα
  #------------------------------------------
  _, Uv, _ = ViscousStrain(Ce, C)
  Cell_ = [get_array(Uv)[:]; λα]  # TODO: Another problem with TensorValue slice
  return true, VectorValue(Cell_)
end


function ViscousDissipation(obj::ViscousIncompressible,
                       Se_::Function, ∂Se∂Ce_::Function,
                       F::TensorValue, Fn::TensorValue, A::VectorValue)
  Uvn = TensorValue{3,3}(A[1:9]...)
  λαn = A[10]
  #------------------------------------------
  # Get kinematics
  #------------------------------------------
  invUvn  = inv(Uvn)
  C = Cauchy(F)
  Cn = Cauchy(Fn)
  Ceᵗʳ = ElasticCauchy(C, invUvn)
  Cen  = ElasticCauchy(Cn, invUvn)
  #------------------------------------------
  # Return mapping algorithm
  #------------------------------------------
  Ce, λα = return_mapping_algorithm!(obj, Se_, ∂Se∂Ce_, C, Ceᵗʳ, Cen, λαn)
  #------------------------------------------
  # Dissipation
  #------------------------------------------
  τ = obj.τ
  Se = Se_(Ce)
  Ge = cof(Ce)
  ∂Se∂Ce = ∂Se∂Ce_(Ce)
  α = 1.e5abs(tr(∂Se∂Ce))  # Ensure invertibility of the elasticity tensor.
  invCCe = inv(2*∂Se∂Ce + α*Ge⊗Ge)
  ∂Se = -1/τ * (Se - λα*Ge)
  Dvis = -Se ⊙ (invCCe ⊙ ∂Se)
  Dvis
end
