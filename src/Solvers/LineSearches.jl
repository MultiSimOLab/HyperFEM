 

abstract type AbstractLineSearch end

struct LineSearch <: AbstractLineSearch
  function LineSearch()
    new()
  end
  function (obj::LineSearch)(x::AbstractVector, dx::AbstractVector, b::AbstractVector, op::NonlinearOperator)
    return 1.0
  end
end

struct Roman_LS <: AbstractLineSearch
  maxiter::Int
  αmin::Float64
  ρ::Float64
  c::Float64
  function Roman_LS(; maxiter::Int64=50, αmin::Float64=1e-16, ρ::Float64=0.5, c::Float64=0.95)
    new(maxiter, αmin, ρ, c)
  end

  function (obj::Roman_LS)(x::AbstractVector, dx::AbstractVector, b::AbstractVector, op::NonlinearOperator)

    maxiter, αmin, ρ, c = obj.maxiter, obj.αmin, obj.ρ, obj.c
    m = 0
    α = 1.0
    R₀ = b' * dx

    while α > αmin && m < maxiter
      residual!(b, op, x + α * dx)
      R = b' * dx
      if R <= c * R₀
        break
      end
      α *= ρ
      m += 1
    end
    @show α
    return α
    # x .+= α * dx
  end
end

struct Injectivity_Preserving_LS{A} <: AbstractLineSearch
  α::CellState
  maxiter::Int
  αmin::Float64
  ρ::Float64
  c::Float64
  caches::A
  function Injectivity_Preserving_LS(α::CellState, U, V; maxiter::Int64=50, αmin::Float64=1e-16, ρ::Float64=0.5, c::Float64=0.95)
    caches = (U, V)
    new{typeof(caches)}(α, maxiter, αmin, ρ, c, caches)
  end


  function InjectivityCheck(α, ∇u, ∇du)
    ε = 1e-6
    F = ∇u + one(∇u)
    J = det(F)
    H = J * inv(F)'
    return true, min(abs((ε - J) / (det(∇du) + tr(H' * ∇du) + ε)), 1.0)
  end



  function (obj::Injectivity_Preserving_LS)(x::AbstractVector, dx::AbstractVector, b::AbstractVector, op::NonlinearOperator)

    αstate, maxiter, αmin, ρ, c = obj.α, obj.maxiter, obj.αmin, obj.ρ, obj.c
    #update cell state
    U, V = obj.caches
    xh = FEFunction(U, x)
    dxh = FEFunction(V, dx)
    update_state!(InjectivityCheck, αstate, ∇(xh)', ∇(dxh)')

    m = 0
    α = minimum(minimum((αstate.values)))
    R₀ = b' * dx

    while α > αmin && m < maxiter
      residual!(b, op, x + α * dx)
      R = b' * dx
      if R <= c * R₀
        break
      end
      α *= ρ
      m += 1
    end
    println("α = ", α)
    return α
    # x .+= α * dx
  end

end

