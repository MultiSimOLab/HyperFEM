
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
  0.5A×A
end


"""
Jacobian regularization
"""
function logreg(J; Threshold=0.01)
  if J >= Threshold
    return log(J)
  else
    return log(Threshold) - (3.0 / 2.0) + (2 / Threshold) * J - (1 / (2 * Threshold^2)) * J^2
  end
end


function _∂H∂F_2D()
  TensorValue(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
end


function trAA(A::TensorValue{3, 3, T, N}) where {T, N}
  @warn "trAA is deprecated and will be removed. Use the double contraction operator : instead." maxlog=1
  return sum(A.data[i]*A.data[i] for i in 1:N)
end


@generated function Tensorize(A::VectorValue{D,Float64}) where {D}
  str = ""
  for i in 1:D
    str *= "A.data[$i], "
  end
  Meta.parse("TensorValue($str)")
end