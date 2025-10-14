
"""
    sqrt(A::TensorValue{3})::TensorValue{3}

Compute the square root of a 3x3 matrix by means of eigen decomposition.
"""
function sqrt(A::TensorValue{3})
  λ, Q = eigen(A)
  λ = sqrt.(λ)
  TensorValue{3}(λ[1]*Q[1:3]*Q[1:3]' + λ[2]*Q[4:6]*Q[4:6]' + λ[3]*Q[7:9]*Q[7:9]')
end


"""
    cof(A::TensorValue)::TensorValue

Calculate the cofactor of a matrix.
"""
function cof(A::TensorValue)
  return det(A)*inv(A')
end


function _∂H∂F_2D()
  TensorValue(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
end


function trAA(A::TensorValue{3, 3, T, N}) where {T, N}
  return sum(A.data[i]*A.data[i] for i in 1:N)
end