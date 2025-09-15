module TensorAlgebra

using Gridap
using Gridap.TensorValues
using StaticArrays
using LinearAlgebra
import Base: *
import Base: +
import Base: sqrt

export (*)
export (×ᵢ⁴)
export (+)
export (⊗₁₂³)
export (⊗₁₃²)
export (⊗₁²³)
export (⊗₁₂³⁴)
export (⊗₁₃²⁴)
export (⊗₁₄²³)
export (⊗₁²)
export I3
export I9
export I2
export I4
export logreg
export Tensorize
export δᵢⱼδₖₗ2D
export δᵢₖδⱼₗ2D
export δᵢₗδⱼₖ2D
export δᵢⱼδₖₗ3D
export δᵢₖδⱼₗ3D
export δᵢₗδⱼₖ3D
export sqrt
export cof
export contraction_IP_JPKL
export contraction_IP_PJKL
 

include("FunctionalAlgebra.jl")
export Box
export Ellipsoid

# outer ⊗ \otimes
# inner ⊙ \odot
# cross × \times
# sum +
# dot ⋅ * 


"""
  sqrt(A::TensorValue{3})::TensorValue{3}

  Compute the square root of a 3x3 matrix by means of eigen decomposition.

  # Arguments
  - `A::TensorValue{3}`: the matrix to calculate the square root

  # Returns
  - `::TensorValue{3}`: the squared root matrix
"""
function sqrt(A::TensorValue{3})
  λ, Q = eigen(get_array(A))  # TODO: the get_array must be removed as long as it is supported after https://github.com/gridap/Gridap.jl/pull/1157
  λ = sqrt.(λ)
  TensorValue{3}(λ[1]*Q[1:3]*Q[1:3]' + λ[2]*Q[4:6]*Q[4:6]' + λ[3]*Q[7:9]*Q[7:9]')
end


"""
  cof(A::TensorValue)::TensorValue

  Calculate the cofactor of a matrix.

  # Arguments
  - `A::TensorValue`: the matrix to calculate.

  # Returns
  - `TensorValue`: the cofactor matrix.
"""
function cof(A::TensorValue)
  return det(A)*inv(A')
end


_flat_idx(i::Int, j::Int, N::Int) = i + N*(j-1)
_flat_idx(i::Int, j::Int, k::Int, l::Int, N::Int) = _flat_idx(_flat_idx(i,j,N), _flat_idx(k,l,N), N)
_full_idx2(α::Int, N::Int) = ((α-1)%N+1 ,(α-1)÷N+1)
_full_idx4(α::Int, β::Int, N::Int) = (_full_idx2(α,N)..., _full_idx2(β,N)...)
_full_idx4(α::Int, N::Int) = _full_idx4(_full_idx2(α,N*N)...,N)

function _Kroneckerδδ(δδ::Function, N::Int)
  TensorValue{N*N,N*N,Float64}(ntuple(α -> begin
    i, j, k, l = _full_idx4(α,N)
    δδ(i,j,k,l) ? 1.0 : 0.0
  end,
  N*N*N*N))
end

const δᵢⱼδₖₗ2D = _Kroneckerδδ((i,j,k,l) -> i==j && k==l, 2)
const δᵢₖδⱼₗ2D = _Kroneckerδδ((i,j,k,l) -> i==k && j==l, 2)
const δᵢₗδⱼₖ2D = _Kroneckerδδ((i,j,k,l) -> i==l && j==k, 2)

const δᵢⱼδₖₗ3D = _Kroneckerδδ((i,j,k,l) -> i==j && k==l, 3)
const δᵢₖδⱼₗ3D = _Kroneckerδδ((i,j,k,l) -> i==k && j==l, 3)
const δᵢₗδⱼₖ3D = _Kroneckerδδ((i,j,k,l) -> i==l && j==k, 3)


function _∂H∂F_2D()
  TensorValue(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
end


function Gridap.TensorValues.outer(A::TensorValue{D,D,Float64}, B::TensorValue{D,D,Float64}) where {D}
  return (A ⊗₁₂³⁴ B)
end

function Gridap.TensorValues.outer(A::VectorValue{D,Float64}, B::VectorValue{D,Float64}) where {D}
  return (A ⊗₁² B)
end


"""
  **`⊗₁²(A::VectorValue{D}, B::VectorValue{D})::TensorValue{D,D}`**

  Outer product of two first-order tensors (vectors), returning a second-order tensor (matrix).
"""
@generated function (⊗₁²)(A::VectorValue{D,Float64}, B::VectorValue{D,Float64}) where {D}
  str = ""
  for iB in 1:D
    for iA in 1:D
      str *= "A.data[$iA] * B.data[$iB], "
    end
  end
  Meta.parse("TensorValue{D,D, Float64}($str)")
end


"""
  **`⊗₁₃²⁴(A::TensorValue{D}, B::TensorValue{D})::TensorValue{D*D}`**

  Outer product of two second-order tensors (matrices), returning a fourth-order tensor 
  represented in a `D² x D²` flattened matrix using combined indices.
"""
@generated function (⊗₁₂³⁴)(A::TensorValue{D,D,Float64}, B::TensorValue{D,D,Float64}) where {D}
  str = ""
  for iB in 1:D*D
    for iA in 1:D*D
      str *= "A.data[$iA] * B.data[$iB], "
    end
  end
  Meta.parse("TensorValue{D*D,D*D, Float64}($str)")
end


"""
  **`⊗₁₃²⁴(A::TensorValue{D}, B::TensorValue{D})::TensorValue{D*D}`**

  Outer product of two second-order tensors (matrices), returning a fourth-order tensor 
  represented in a `D² x D²` flattened matrix using combined indices.
"""
@generated function (⊗₁₃²⁴)(A::TensorValue{D}, B::TensorValue{D}) where D
  str = ""
  for l in 1:D
    for k in 1:D
      for j in 1:D
        for i in 1:D
          str *= "A[$i,$k]*B[$j,$l],"
        end
      end
    end
  end
  Meta.parse("TensorValue{D*D}($str)")
end


"""
  **`⊗₁₄²³(A::TensorValue{D}, B::TensorValue{D})::TensorValue{D*D}`**

  Outer product of two second-order tensors (matrices), returning a fourth-order tensor 
  represented in a `D² x D²` flattened matrix using combined indices.
"""
@generated function (⊗₁₄²³)(A::TensorValue{D}, B::TensorValue{D}) where D
  str = ""
  for l in 1:D
    for k in 1:D
      for j in 1:D
        for i in 1:D
          str *= "A[$i,$l]*B[$j,$k],"
        end
      end
    end
  end
  Meta.parse("TensorValue{D*D}($str)")
end


"""
  **`⊗₁²³(A::VectorValue{D}, B::TensorValue{D})::TensorValue{D,D*D}`**

  Outer product of a first-order and second-order tensors (vector and matrix),
  returning a third-order tensor represented in a `D x D²` flattened matrix using combined indices.
"""
@generated function (⊗₁²³)(V::VectorValue{D,Float64}, A::TensorValue{D,D,Float64}) where {D}
  str = ""
  for iA in 1:D*D
    for iV in 1:D
      str *= "A.data[$iA] * V.data[$iV], "
    end
  end
  Meta.parse("TensorValue{D,D*D, Float64, D*D*D}($str)")
end


"""
  **`⊗₁²³(A::TensorValue{D}, B::VectorValue{D})::TensorValue{D,D*D}`**

  Outer product of a second-order and first-order tensors (matrix and vector),
  returning a third-order tensor represented in a `D x D²` flattened matrix using combined indices.
"""
@generated function (⊗₁₂³)(A::TensorValue{D,D,Float64}, V::VectorValue{D,Float64}) where {D}
  str = ""
  for iV in 1:D
    for iA in 1:D*D
      str *= "A.data[$iA] * V.data[$iV], "
    end
  end
  Meta.parse("TensorValue{D,D*D, Float64,D*D*D}($str)")
end


"""
  **`⊗₁²³(A::TensorValue{D}, B::TensorValue{D})::TensorValue{D,D*D}`**

  Outer product of a second-order and first-order tensors (matrix and vector),
  returning a third-order tensor represented in a `D x D²` flattened matrix using combined indices.
"""
@generated function (⊗₁₃²)(A::TensorValue{D}, V::VectorValue{D}) where D
  str = ""
  for k in 1:D
    for j in 1:D
      for i in 1:D
        str *= "A[$i,$k]*V[$j],"
      end
    end
  end
  Meta.parse("TensorValue{D,D*D}($str)")
end


function (×ᵢ⁴)(A::TensorValue{3,3,Float64})

  TensorValue(0.0, 0.0, 0.0, 0.0, A[9], -A[8], 0.0, -A[6], A[5], 0.0, 0.0, 0.0, -A[9],
    0.0, A[7], A[6], 0.0, -A[4], 0.0, 0.0, 0.0, A[8], -A[7], 0.0, -A[5], A[4], 0.0, 0.0, -A[9],
    A[8], 0.0, 0.0, 0.0, 0.0, A[3], -A[2], A[9], 0.0, -A[7], 0.0, 0.0, 0.0, -A[3], 0.0,
    A[1], -A[8], A[7], 0.0, 0.0, 0.0, 0.0, A[2], -A[1], 0.0, 0.0, A[6], -A[5], 0.0,
    -A[3], A[2], 0.0, 0.0, 0.0, -A[6], 0.0, A[4], A[3], 0.0, -A[1],
    0.0, 0.0, 0.0, A[5], -A[4], 0.0, -A[2], A[1], 0.0, 0.0, 0.0, 0.0)
end


function Gridap.TensorValues.cross(A::TensorValue{3,3,T1}, B::TensorValue{3,3,T2}) where {T1,T2}

  TensorValue(A[5] * B[9] - A[6] * B[8] - A[8] * B[6] + A[9] * B[5],
    A[6] * B[7] - A[4] * B[9] + A[7] * B[6] - A[9] * B[4],
    A[4] * B[8] - A[5] * B[7] - A[7] * B[5] + A[8] * B[4],
    A[3] * B[8] - A[2] * B[9] + A[8] * B[3] - A[9] * B[2],
    A[1] * B[9] - A[3] * B[7] - A[7] * B[3] + A[9] * B[1],
    A[2] * B[7] - A[1] * B[8] + A[7] * B[2] - A[8] * B[1],
    A[2] * B[6] - A[3] * B[5] - A[5] * B[3] + A[6] * B[2],
    A[3] * B[4] - A[1] * B[6] + A[4] * B[3] - A[6] * B[1],
    A[1] * B[5] - A[2] * B[4] - A[4] * B[2] + A[5] * B[1])
end


function Gridap.TensorValues.cross(H::TensorValue{9,9,T1}, A::TensorValue{3,3,T2}) where {T1,T2}

  TensorValue(A[9] * H[37] - A[8] * H[46] - A[6] * H[64] + A[5] * H[73],
    A[9] * H[38] - A[8] * H[47] - A[6] * H[65] + A[5] * H[74],
    A[9] * H[39] - A[8] * H[48] - A[6] * H[66] + A[5] * H[75],
    A[9] * H[40] - A[8] * H[49] - A[6] * H[67] + A[5] * H[76],
    A[9] * H[41] - A[8] * H[50] - A[6] * H[68] + A[5] * H[77],
    A[9] * H[42] - A[8] * H[51] - A[6] * H[69] + A[5] * H[78],
    A[9] * H[43] - A[8] * H[52] - A[6] * H[70] + A[5] * H[79],
    A[9] * H[44] - A[8] * H[53] - A[6] * H[71] + A[5] * H[80],
    A[9] * H[45] - A[8] * H[54] - A[6] * H[72] + A[5] * H[81],
    A[7] * H[46] - A[9] * H[28] + A[6] * H[55] - A[4] * H[73],
    A[7] * H[47] - A[9] * H[29] + A[6] * H[56] - A[4] * H[74],
    A[7] * H[48] - A[9] * H[30] + A[6] * H[57] - A[4] * H[75],
    A[7] * H[49] - A[9] * H[31] + A[6] * H[58] - A[4] * H[76],
    A[7] * H[50] - A[9] * H[32] + A[6] * H[59] - A[4] * H[77],
    A[7] * H[51] - A[9] * H[33] + A[6] * H[60] - A[4] * H[78],
    A[7] * H[52] - A[9] * H[34] + A[6] * H[61] - A[4] * H[79],
    A[7] * H[53] - A[9] * H[35] + A[6] * H[62] - A[4] * H[80],
    A[7] * H[54] - A[9] * H[36] + A[6] * H[63] - A[4] * H[81],
    A[8] * H[28] - A[7] * H[37] - A[5] * H[55] + A[4] * H[64],
    A[8] * H[29] - A[7] * H[38] - A[5] * H[56] + A[4] * H[65],
    A[8] * H[30] - A[7] * H[39] - A[5] * H[57] + A[4] * H[66],
    A[8] * H[31] - A[7] * H[40] - A[5] * H[58] + A[4] * H[67],
    A[8] * H[32] - A[7] * H[41] - A[5] * H[59] + A[4] * H[68],
    A[8] * H[33] - A[7] * H[42] - A[5] * H[60] + A[4] * H[69],
    A[8] * H[34] - A[7] * H[43] - A[5] * H[61] + A[4] * H[70],
    A[8] * H[35] - A[7] * H[44] - A[5] * H[62] + A[4] * H[71],
    A[8] * H[36] - A[7] * H[45] - A[5] * H[63] + A[4] * H[72],
    A[8] * H[19] - A[9] * H[10] + A[3] * H[64] - A[2] * H[73],
    A[8] * H[20] - A[9] * H[11] + A[3] * H[65] - A[2] * H[74],
    A[8] * H[21] - A[9] * H[12] + A[3] * H[66] - A[2] * H[75],
    A[8] * H[22] - A[9] * H[13] + A[3] * H[67] - A[2] * H[76],
    A[8] * H[23] - A[9] * H[14] + A[3] * H[68] - A[2] * H[77],
    A[8] * H[24] - A[9] * H[15] + A[3] * H[69] - A[2] * H[78],
    A[8] * H[25] - A[9] * H[16] + A[3] * H[70] - A[2] * H[79],
    A[8] * H[26] - A[9] * H[17] + A[3] * H[71] - A[2] * H[80],
    A[8] * H[27] - A[9] * H[18] + A[3] * H[72] - A[2] * H[81],
    A[9] * H[1] - A[7] * H[19] - A[3] * H[55] + A[1] * H[73],
    A[9] * H[2] - A[7] * H[20] - A[3] * H[56] + A[1] * H[74],
    A[9] * H[3] - A[7] * H[21] - A[3] * H[57] + A[1] * H[75],
    A[9] * H[4] - A[7] * H[22] - A[3] * H[58] + A[1] * H[76],
    A[9] * H[5] - A[7] * H[23] - A[3] * H[59] + A[1] * H[77],
    A[9] * H[6] - A[7] * H[24] - A[3] * H[60] + A[1] * H[78],
    A[9] * H[7] - A[7] * H[25] - A[3] * H[61] + A[1] * H[79],
    A[9] * H[8] - A[7] * H[26] - A[3] * H[62] + A[1] * H[80],
    A[9] * H[9] - A[7] * H[27] - A[3] * H[63] + A[1] * H[81],
    A[7] * H[10] - A[8] * H[1] + A[2] * H[55] - A[1] * H[64],
    A[7] * H[11] - A[8] * H[2] + A[2] * H[56] - A[1] * H[65],
    A[7] * H[12] - A[8] * H[3] + A[2] * H[57] - A[1] * H[66],
    A[7] * H[13] - A[8] * H[4] + A[2] * H[58] - A[1] * H[67],
    A[7] * H[14] - A[8] * H[5] + A[2] * H[59] - A[1] * H[68],
    A[7] * H[15] - A[8] * H[6] + A[2] * H[60] - A[1] * H[69],
    A[7] * H[16] - A[8] * H[7] + A[2] * H[61] - A[1] * H[70],
    A[7] * H[17] - A[8] * H[8] + A[2] * H[62] - A[1] * H[71],
    A[7] * H[18] - A[8] * H[9] + A[2] * H[63] - A[1] * H[72],
    A[6] * H[10] - A[5] * H[19] - A[3] * H[37] + A[2] * H[46],
    A[6] * H[11] - A[5] * H[20] - A[3] * H[38] + A[2] * H[47],
    A[6] * H[12] - A[5] * H[21] - A[3] * H[39] + A[2] * H[48],
    A[6] * H[13] - A[5] * H[22] - A[3] * H[40] + A[2] * H[49],
    A[6] * H[14] - A[5] * H[23] - A[3] * H[41] + A[2] * H[50],
    A[6] * H[15] - A[5] * H[24] - A[3] * H[42] + A[2] * H[51],
    A[6] * H[16] - A[5] * H[25] - A[3] * H[43] + A[2] * H[52],
    A[6] * H[17] - A[5] * H[26] - A[3] * H[44] + A[2] * H[53],
    A[6] * H[18] - A[5] * H[27] - A[3] * H[45] + A[2] * H[54],
    A[4] * H[19] - A[6] * H[1] + A[3] * H[28] - A[1] * H[46],
    A[4] * H[20] - A[6] * H[2] + A[3] * H[29] - A[1] * H[47],
    A[4] * H[21] - A[6] * H[3] + A[3] * H[30] - A[1] * H[48],
    A[4] * H[22] - A[6] * H[4] + A[3] * H[31] - A[1] * H[49],
    A[4] * H[23] - A[6] * H[5] + A[3] * H[32] - A[1] * H[50],
    A[4] * H[24] - A[6] * H[6] + A[3] * H[33] - A[1] * H[51],
    A[4] * H[25] - A[6] * H[7] + A[3] * H[34] - A[1] * H[52],
    A[4] * H[26] - A[6] * H[8] + A[3] * H[35] - A[1] * H[53],
    A[4] * H[27] - A[6] * H[9] + A[3] * H[36] - A[1] * H[54],
    A[5] * H[1] - A[4] * H[10] - A[2] * H[28] + A[1] * H[37],
    A[5] * H[2] - A[4] * H[11] - A[2] * H[29] + A[1] * H[38],
    A[5] * H[3] - A[4] * H[12] - A[2] * H[30] + A[1] * H[39],
    A[5] * H[4] - A[4] * H[13] - A[2] * H[31] + A[1] * H[40],
    A[5] * H[5] - A[4] * H[14] - A[2] * H[32] + A[1] * H[41],
    A[5] * H[6] - A[4] * H[15] - A[2] * H[33] + A[1] * H[42],
    A[5] * H[7] - A[4] * H[16] - A[2] * H[34] + A[1] * H[43],
    A[5] * H[8] - A[4] * H[17] - A[2] * H[35] + A[1] * H[44],
    A[5] * H[9] - A[4] * H[18] - A[2] * H[36] + A[1] * H[45])
end

function Gridap.TensorValues.cross(A::TensorValue{3,3,T1}, H::TensorValue{9,9,T2}) where {T1,T2}

  TensorValue(A[5] * H[9] - A[6] * H[8] - A[8] * H[6] + A[9] * H[5],
    A[6] * H[7] - A[4] * H[9] + A[7] * H[6] - A[9] * H[4],
    A[4] * H[8] - A[5] * H[7] - A[7] * H[5] + A[8] * H[4],
    A[3] * H[8] - A[2] * H[9] + A[8] * H[3] - A[9] * H[2],
    A[1] * H[9] - A[3] * H[7] - A[7] * H[3] + A[9] * H[1],
    A[2] * H[7] - A[1] * H[8] + A[7] * H[2] - A[8] * H[1],
    A[2] * H[6] - A[3] * H[5] - A[5] * H[3] + A[6] * H[2],
    A[3] * H[4] - A[1] * H[6] + A[4] * H[3] - A[6] * H[1],
    A[1] * H[5] - A[2] * H[4] - A[4] * H[2] + A[5] * H[1],
    A[5] * H[18] - A[6] * H[17] - A[8] * H[15] + A[9] * H[14],
    A[6] * H[16] - A[4] * H[18] + A[7] * H[15] - A[9] * H[13],
    A[4] * H[17] - A[5] * H[16] - A[7] * H[14] + A[8] * H[13],
    A[3] * H[17] - A[2] * H[18] + A[8] * H[12] - A[9] * H[11],
    A[1] * H[18] - A[3] * H[16] - A[7] * H[12] + A[9] * H[10],
    A[2] * H[16] - A[1] * H[17] + A[7] * H[11] - A[8] * H[10],
    A[2] * H[15] - A[3] * H[14] - A[5] * H[12] + A[6] * H[11],
    A[3] * H[13] - A[1] * H[15] + A[4] * H[12] - A[6] * H[10],
    A[1] * H[14] - A[2] * H[13] - A[4] * H[11] + A[5] * H[10],
    A[5] * H[27] - A[6] * H[26] - A[8] * H[24] + A[9] * H[23],
    A[6] * H[25] - A[4] * H[27] + A[7] * H[24] - A[9] * H[22],
    A[4] * H[26] - A[5] * H[25] - A[7] * H[23] + A[8] * H[22],
    A[3] * H[26] - A[2] * H[27] + A[8] * H[21] - A[9] * H[20],
    A[1] * H[27] - A[3] * H[25] - A[7] * H[21] + A[9] * H[19],
    A[2] * H[25] - A[1] * H[26] + A[7] * H[20] - A[8] * H[19],
    A[2] * H[24] - A[3] * H[23] - A[5] * H[21] + A[6] * H[20],
    A[3] * H[22] - A[1] * H[24] + A[4] * H[21] - A[6] * H[19],
    A[1] * H[23] - A[2] * H[22] - A[4] * H[20] + A[5] * H[19],
    A[5] * H[36] - A[6] * H[35] - A[8] * H[33] + A[9] * H[32],
    A[6] * H[34] - A[4] * H[36] + A[7] * H[33] - A[9] * H[31],
    A[4] * H[35] - A[5] * H[34] - A[7] * H[32] + A[8] * H[31],
    A[3] * H[35] - A[2] * H[36] + A[8] * H[30] - A[9] * H[29],
    A[1] * H[36] - A[3] * H[34] - A[7] * H[30] + A[9] * H[28],
    A[2] * H[34] - A[1] * H[35] + A[7] * H[29] - A[8] * H[28],
    A[2] * H[33] - A[3] * H[32] - A[5] * H[30] + A[6] * H[29],
    A[3] * H[31] - A[1] * H[33] + A[4] * H[30] - A[6] * H[28],
    A[1] * H[32] - A[2] * H[31] - A[4] * H[29] + A[5] * H[28],
    A[5] * H[45] - A[6] * H[44] - A[8] * H[42] + A[9] * H[41],
    A[6] * H[43] - A[4] * H[45] + A[7] * H[42] - A[9] * H[40],
    A[4] * H[44] - A[5] * H[43] - A[7] * H[41] + A[8] * H[40],
    A[3] * H[44] - A[2] * H[45] + A[8] * H[39] - A[9] * H[38],
    A[1] * H[45] - A[3] * H[43] - A[7] * H[39] + A[9] * H[37],
    A[2] * H[43] - A[1] * H[44] + A[7] * H[38] - A[8] * H[37],
    A[2] * H[42] - A[3] * H[41] - A[5] * H[39] + A[6] * H[38],
    A[3] * H[40] - A[1] * H[42] + A[4] * H[39] - A[6] * H[37],
    A[1] * H[41] - A[2] * H[40] - A[4] * H[38] + A[5] * H[37],
    A[5] * H[54] - A[6] * H[53] - A[8] * H[51] + A[9] * H[50],
    A[6] * H[52] - A[4] * H[54] + A[7] * H[51] - A[9] * H[49],
    A[4] * H[53] - A[5] * H[52] - A[7] * H[50] + A[8] * H[49],
    A[3] * H[53] - A[2] * H[54] + A[8] * H[48] - A[9] * H[47],
    A[1] * H[54] - A[3] * H[52] - A[7] * H[48] + A[9] * H[46],
    A[2] * H[52] - A[1] * H[53] + A[7] * H[47] - A[8] * H[46],
    A[2] * H[51] - A[3] * H[50] - A[5] * H[48] + A[6] * H[47],
    A[3] * H[49] - A[1] * H[51] + A[4] * H[48] - A[6] * H[46],
    A[1] * H[50] - A[2] * H[49] - A[4] * H[47] + A[5] * H[46],
    A[5] * H[63] - A[6] * H[62] - A[8] * H[60] + A[9] * H[59],
    A[6] * H[61] - A[4] * H[63] + A[7] * H[60] - A[9] * H[58],
    A[4] * H[62] - A[5] * H[61] - A[7] * H[59] + A[8] * H[58],
    A[3] * H[62] - A[2] * H[63] + A[8] * H[57] - A[9] * H[56],
    A[1] * H[63] - A[3] * H[61] - A[7] * H[57] + A[9] * H[55],
    A[2] * H[61] - A[1] * H[62] + A[7] * H[56] - A[8] * H[55],
    A[2] * H[60] - A[3] * H[59] - A[5] * H[57] + A[6] * H[56],
    A[3] * H[58] - A[1] * H[60] + A[4] * H[57] - A[6] * H[55],
    A[1] * H[59] - A[2] * H[58] - A[4] * H[56] + A[5] * H[55],
    A[5] * H[72] - A[6] * H[71] - A[8] * H[69] + A[9] * H[68],
    A[6] * H[70] - A[4] * H[72] + A[7] * H[69] - A[9] * H[67],
    A[4] * H[71] - A[5] * H[70] - A[7] * H[68] + A[8] * H[67],
    A[3] * H[71] - A[2] * H[72] + A[8] * H[66] - A[9] * H[65],
    A[1] * H[72] - A[3] * H[70] - A[7] * H[66] + A[9] * H[64],
    A[2] * H[70] - A[1] * H[71] + A[7] * H[65] - A[8] * H[64],
    A[2] * H[69] - A[3] * H[68] - A[5] * H[66] + A[6] * H[65],
    A[3] * H[67] - A[1] * H[69] + A[4] * H[66] - A[6] * H[64],
    A[1] * H[68] - A[2] * H[67] - A[4] * H[65] + A[5] * H[64],
    A[5] * H[81] - A[6] * H[80] - A[8] * H[78] + A[9] * H[77],
    A[6] * H[79] - A[4] * H[81] + A[7] * H[78] - A[9] * H[76],
    A[4] * H[80] - A[5] * H[79] - A[7] * H[77] + A[8] * H[76],
    A[3] * H[80] - A[2] * H[81] + A[8] * H[75] - A[9] * H[74],
    A[1] * H[81] - A[3] * H[79] - A[7] * H[75] + A[9] * H[73],
    A[2] * H[79] - A[1] * H[80] + A[7] * H[74] - A[8] * H[73],
    A[2] * H[78] - A[3] * H[77] - A[5] * H[75] + A[6] * H[74],
    A[3] * H[76] - A[1] * H[78] + A[4] * H[75] - A[6] * H[73],
    A[1] * H[77] - A[2] * H[76] - A[4] * H[74] + A[5] * H[73])
end

function Gridap.TensorValues.cross(A::TensorValue{3,9,T1}, B::TensorValue{3,3,T2}) where {T1,T2}

  TensorValue{3,9,Float64,27}(A[13] * B[9] - A[16] * B[8] - A[22] * B[6] + A[25] * B[5],
    A[14] * B[9] - A[17] * B[8] - A[23] * B[6] + A[26] * B[5],
    A[15] * B[9] - A[18] * B[8] - A[24] * B[6] + A[27] * B[5],
    A[16] * B[7] - A[10] * B[9] + A[19] * B[6] - A[25] * B[4],
    A[17] * B[7] - A[11] * B[9] + A[20] * B[6] - A[26] * B[4],
    A[18] * B[7] - A[12] * B[9] + A[21] * B[6] - A[27] * B[4],
    A[10] * B[8] - A[13] * B[7] - A[19] * B[5] + A[22] * B[4],
    A[11] * B[8] - A[14] * B[7] - A[20] * B[5] + A[23] * B[4],
    A[12] * B[8] - A[15] * B[7] - A[21] * B[5] + A[24] * B[4],
    A[7] * B[8] - A[4] * B[9] + A[22] * B[3] - A[25] * B[2],
    A[8] * B[8] - A[5] * B[9] + A[23] * B[3] - A[26] * B[2],
    A[9] * B[8] - A[6] * B[9] + A[24] * B[3] - A[27] * B[2],
    A[1] * B[9] - A[7] * B[7] - A[19] * B[3] + A[25] * B[1],
    A[2] * B[9] - A[8] * B[7] - A[20] * B[3] + A[26] * B[1],
    A[3] * B[9] - A[9] * B[7] - A[21] * B[3] + A[27] * B[1],
    A[4] * B[7] - A[1] * B[8] + A[19] * B[2] - A[22] * B[1],
    A[5] * B[7] - A[2] * B[8] + A[20] * B[2] - A[23] * B[1],
    A[6] * B[7] - A[3] * B[8] + A[21] * B[2] - A[24] * B[1],
    A[4] * B[6] - A[7] * B[5] - A[13] * B[3] + A[16] * B[2],
    A[5] * B[6] - A[8] * B[5] - A[14] * B[3] + A[17] * B[2],
    A[6] * B[6] - A[9] * B[5] - A[15] * B[3] + A[18] * B[2],
    A[7] * B[4] - A[1] * B[6] + A[10] * B[3] - A[16] * B[1],
    A[8] * B[4] - A[2] * B[6] + A[11] * B[3] - A[17] * B[1],
    A[9] * B[4] - A[3] * B[6] + A[12] * B[3] - A[18] * B[1],
    A[1] * B[5] - A[4] * B[4] - A[10] * B[2] + A[13] * B[1],
    A[2] * B[5] - A[5] * B[4] - A[11] * B[2] + A[14] * B[1],
    A[3] * B[5] - A[6] * B[4] - A[12] * B[2] + A[15] * B[1])
end

function Gridap.TensorValues.cross(A::SMatrix, B::SMatrix)
  return get_array(TensorValue(A) × TensorValue(B))
end

function Gridap.TensorValues.outer(A::SVector, B::SVector)
  return get_array(VectorValue(A) ⊗ VectorValue(B))
end

function Gridap.TensorValues.inner(Ten1::TensorValue{9,9,Float64}, Ten2::TensorValue{3,3,Float64})
  TensorValue(Ten1[1] * Ten2[1] + Ten1[10] * Ten2[2] + Ten1[19] * Ten2[3] + Ten1[28] * Ten2[4] + Ten1[37] * Ten2[5] + Ten1[46] * Ten2[6] + Ten1[55] * Ten2[7] + Ten1[64] * Ten2[8] + Ten1[73] * Ten2[9],
    Ten1[2] * Ten2[1] + Ten1[11] * Ten2[2] + Ten1[20] * Ten2[3] + Ten1[29] * Ten2[4] + Ten1[38] * Ten2[5] + Ten1[47] * Ten2[6] + Ten1[56] * Ten2[7] + Ten1[65] * Ten2[8] + Ten1[74] * Ten2[9],
    Ten1[3] * Ten2[1] + Ten1[12] * Ten2[2] + Ten1[21] * Ten2[3] + Ten1[30] * Ten2[4] + Ten1[39] * Ten2[5] + Ten1[48] * Ten2[6] + Ten1[57] * Ten2[7] + Ten1[66] * Ten2[8] + Ten1[75] * Ten2[9],
    Ten1[4] * Ten2[1] + Ten1[13] * Ten2[2] + Ten1[22] * Ten2[3] + Ten1[31] * Ten2[4] + Ten1[40] * Ten2[5] + Ten1[49] * Ten2[6] + Ten1[58] * Ten2[7] + Ten1[67] * Ten2[8] + Ten1[76] * Ten2[9],
    Ten1[5] * Ten2[1] + Ten1[14] * Ten2[2] + Ten1[23] * Ten2[3] + Ten1[32] * Ten2[4] + Ten1[41] * Ten2[5] + Ten1[50] * Ten2[6] + Ten1[59] * Ten2[7] + Ten1[68] * Ten2[8] + Ten1[77] * Ten2[9],
    Ten1[6] * Ten2[1] + Ten1[15] * Ten2[2] + Ten1[24] * Ten2[3] + Ten1[33] * Ten2[4] + Ten1[42] * Ten2[5] + Ten1[51] * Ten2[6] + Ten1[60] * Ten2[7] + Ten1[69] * Ten2[8] + Ten1[78] * Ten2[9],
    Ten1[7] * Ten2[1] + Ten1[16] * Ten2[2] + Ten1[25] * Ten2[3] + Ten1[34] * Ten2[4] + Ten1[43] * Ten2[5] + Ten1[52] * Ten2[6] + Ten1[61] * Ten2[7] + Ten1[70] * Ten2[8] + Ten1[79] * Ten2[9],
    Ten1[8] * Ten2[1] + Ten1[17] * Ten2[2] + Ten1[26] * Ten2[3] + Ten1[35] * Ten2[4] + Ten1[44] * Ten2[5] + Ten1[53] * Ten2[6] + Ten1[62] * Ten2[7] + Ten1[71] * Ten2[8] + Ten1[80] * Ten2[9],
    Ten1[9] * Ten2[1] + Ten1[18] * Ten2[2] + Ten1[27] * Ten2[3] + Ten1[36] * Ten2[4] + Ten1[45] * Ten2[5] + Ten1[54] * Ten2[6] + Ten1[63] * Ten2[7] + Ten1[72] * Ten2[8] + Ten1[81] * Ten2[9])
end

function Gridap.TensorValues.inner(Ten1::TensorValue{3,9,Float64}, Ten2::TensorValue{3,3,Float64})
  VectorValue(Ten1[1] * Ten2[1] + Ten1[4] * Ten2[2] + Ten1[7] * Ten2[3] + Ten1[10] * Ten2[4] + Ten1[13] * Ten2[5] + Ten1[16] * Ten2[6] + Ten1[19] * Ten2[7] + Ten1[22] * Ten2[8] + Ten1[25] * Ten2[9],
    Ten1[2] * Ten2[1] + Ten1[5] * Ten2[2] + Ten1[8] * Ten2[3] + Ten1[11] * Ten2[4] + Ten1[14] * Ten2[5] + Ten1[17] * Ten2[6] + Ten1[20] * Ten2[7] + Ten1[23] * Ten2[8] + Ten1[26] * Ten2[9],
    Ten1[3] * Ten2[1] + Ten1[6] * Ten2[2] + Ten1[9] * Ten2[3] + Ten1[12] * Ten2[4] + Ten1[15] * Ten2[5] + Ten1[18] * Ten2[6] + Ten1[21] * Ten2[7] + Ten1[24] * Ten2[8] + Ten1[27] * Ten2[9])
end

function Gridap.TensorValues.inner(Ten1::TensorValue{2,4,Float64}, Ten2::TensorValue{2,2,Float64})
  VectorValue(Ten1[1] * Ten2[1] + Ten1[3] * Ten2[2] + Ten1[5] * Ten2[3] + Ten1[7] * Ten2[4],
    Ten1[2] * Ten2[1] + Ten1[4] * Ten2[2] + Ten1[6] * Ten2[3] + Ten1[8] * Ten2[4])
end

function Gridap.TensorValues.inner(Ten1::TensorValue{3,9,Float64}, Ten2::VectorValue{3,Float64})
  TensorValue(Ten1[1] * Ten2[1] + Ten1[10] * Ten2[2] + Ten1[19] * Ten2[3],
    Ten1[2] * Ten2[1] + Ten1[11] * Ten2[2] + Ten1[20] * Ten2[3],
    Ten1[3] * Ten2[1] + Ten1[12] * Ten2[2] + Ten1[21] * Ten2[3],
    Ten1[4] * Ten2[1] + Ten1[13] * Ten2[2] + Ten1[22] * Ten2[3],
    Ten1[5] * Ten2[1] + Ten1[14] * Ten2[2] + Ten1[23] * Ten2[3],
    Ten1[6] * Ten2[1] + Ten1[15] * Ten2[2] + Ten1[24] * Ten2[3],
    Ten1[7] * Ten2[1] + Ten1[16] * Ten2[2] + Ten1[25] * Ten2[3],
    Ten1[8] * Ten2[1] + Ten1[17] * Ten2[2] + Ten1[26] * Ten2[3],
    Ten1[9] * Ten2[1] + Ten1[18] * Ten2[2] + Ten1[27] * Ten2[3])
end

function Gridap.TensorValues.inner(Ten1::TensorValue{2,4,Float64}, Ten2::VectorValue{2,Float64})
  TensorValue(Ten1[1] * Ten2[1] + Ten1[5] * Ten2[2],
    Ten1[2] * Ten2[1] + Ten1[6] * Ten2[2],
    Ten1[3] * Ten2[1] + Ten1[7] * Ten2[2],
    Ten1[4] * Ten2[1] + Ten1[8] * Ten2[2])
end

function Gridap.TensorValues.inner(Ten1::TensorValue{4,4,Float64}, Ten2::TensorValue{2,2,Float64})
  TensorValue(Ten1[1] * Ten2[1] + Ten1[5] * Ten2[2] + Ten1[9] * Ten2[3] + Ten1[13] * Ten2[4],
    Ten1[2] * Ten2[1] + Ten1[6] * Ten2[2] + Ten1[10] * Ten2[3] + Ten1[14] * Ten2[4],
    Ten1[3] * Ten2[1] + Ten1[7] * Ten2[2] + Ten1[11] * Ten2[3] + Ten1[15] * Ten2[4],
    Ten1[4] * Ten2[1] + Ten1[8] * Ten2[2] + Ten1[12] * Ten2[3] + Ten1[16] * Ten2[4])
end

function Gridap.TensorValues.inner(Vec1::VectorValue, Ten1::TensorValue)
  return TensorValue(Vec1.data) ⊙ Ten1
end

function (*)(Ten1::TensorValue, Ten2::VectorValue)
  return (⋅)(Ten1, Ten2)
end

function (*)(Ten1::TensorValue, Ten2::TensorValue)
  return (⋅)(Ten1, Ten2)
end


@generated function (+)(A::TensorValue{D,D,Float64}, B::TensorValue{D,D,Float64}) where {D}
  str = ""
  for i in 1:D*D
    str *= "A.data[$i] + B.data[$i], "
  end
  Meta.parse("TensorValue{D,D, Float64}($str)")
end


# Identity matrix
const I_(N) = TensorValue{N,N,Float64}(ntuple(α -> begin
  i,j = _full_idx2(α,N)
  i==j ? 1.0 : 0.0
end,N*N))

const I2 = I_(2)
const I3 = I_(3)
const I4 = I_(4)
const I9 = I_(9)


# Jacobian regularization
function logreg(J; Threshold=0.01)
  if J >= Threshold
    return log(J)
  else
    return log(Threshold) - (3.0 / 2.0) + (2 / Threshold) * J - (1 / (2 * Threshold^2)) * J^2
  end
end

@generated function Tensorize(A::VectorValue{D,Float64}) where {D}
  str = ""
  for i in 1:D
    str *= "A.data[$i], "
  end
  Meta.parse("TensorValue($str)")
end


"""
  **`contraction_IP_PJKL(A::TensorValue{D}, H::TensorValue{D*D})::TensorValue{D*D}`**

  Performs a tensor contraction between a second-order tensor (of size `D × D`)
  and a fourth-order tensor (represented as a `D² × D²` matrix in flattened index notation).
  The operation follows the **index contraction pattern**, where addition is performed for repeated indices.
"""
@generated function contraction_IP_PJKL(A::TensorValue{D}, H::TensorValue{D²}) where {D, D²}
  @assert D*D == D² "Second and Fourth-order tensors size mismatch"
  str = ""
  for l in 1:D
    for k in 1:D
      for j in 1:D
        for i in 1:D
          for p in 1:D
            a = _flat_idx(p,j,D)
            b = _flat_idx(k,l,D)
            str *= "+A[$i,$p]*H[$a,$b]"
          end
          str *= ","
        end
      end
    end
  end
  Meta.parse("TensorValue{D²,D², Float64}($str)")
end


"""
  **`contraction_IP_JPKL(A::TensorValue{D}, H::TensorValue{D*D})::TensorValue{D*D}`**

  Performs a tensor contraction between a second-order tensor (of size `D × D`)
  and a fourth-order tensor (represented as a `D² × D²` matrix in flattened index notation).
  The operation follows the **index contraction pattern**, where addition is performed for repeated indices.
"""
@generated function contraction_IP_JPKL(A::TensorValue{D}, H::TensorValue{D²}) where {D, D²}
  @assert D*D == D² "Second and Fourth-order tensors size mismatch"
  str = ""
  for l in 1:D
    for k in 1:D
      for j in 1:D
        for i in 1:D
          for p in 1:D
            a = _flat_idx(j,p,D)
            b = _flat_idx(k,l,D)
            str *= "+A[$i,$p]*H[$a,$b]"
          end
          str *= ","
        end
      end
    end
  end
  Meta.parse("TensorValue{D²,D², Float64}($str)")
end


end