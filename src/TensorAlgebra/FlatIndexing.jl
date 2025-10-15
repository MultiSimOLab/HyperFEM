
"""Return the linear index of a N-dimensional tensor"""
_flat_idx(i::Int, j::Int, N::Int) = i + N*(j-1)
_flat_idx(i::Int, j::Int, k::Int, l::Int, N::Int) = _flat_idx(_flat_idx(i,j,N), _flat_idx(k,l,N), N*N)

"""Return the cartesian indices of an N-dimensional second-order tensor"""
_full_idx2(α::Int, N::Int) = ((α-1)%N+1 ,(α-1)÷N+1)

"""Return the cartesian indices of an N-dimensional fourth-order tensor"""
_full_idx4(α::Int, β::Int, N::Int) = (_full_idx2(α,N)..., _full_idx2(β,N)...)
_full_idx4(α::Int, N::Int) = _full_idx4(_full_idx2(α,N*N)...,N)
