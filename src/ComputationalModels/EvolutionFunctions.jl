
"The evolution functions have been designed to apply variable boundary conditions."
module EvolutionFunctions

export ramp
export triangular
export step
export sigmoid
export constant

"Return a bounded ramp function from 0 to 1. By default, it is the identity. Otherwise, the scaling factor is 1/T."
function ramp(T::Float64=1.0)
  t::Float64 -> max(min(t/T, 1.0), 0.0)
end

"Return a triangular evolution function ranging from 0 to 1, centered at T, having edges at 0 and 2T."
function triangular(T::Float64)
  triangular(0.0, T)
end

"Return a triangular evolution function ranging from 0 to 1, centered at Tmax, having edges at T0 and 2Tmax-T0."
function triangular(T0::Float64, Tmax::Float64)
  t::Float64 -> begin
    Δ = Tmax - T0
    u = (t - T0) / Δ
    v = (t - Tmax) / Δ
    max(min(u, 1.0-v), 0.0)
  end
end

"Return the Heaviside function."
function step(T::Float64)
  t::Float64 -> t > T ? 1.0 : 0.0
end

"Return a sigmoid-like function with edges at 0 and 2ϵ."
function smoothstep(ϵ::Float64)
  smoothstep(ϵ, ϵ)    
end

"Return a sigmoid-like function centered at T and edges at ±ϵ."
function smoothstep(T::Float64, ϵ::Float64)
  t::Float64 -> begin
    u::Float64 = (t - T + ϵ) / (2 * ϵ)
    if u < 0.0 return 0.0
    elseif u < 1.0 return 3*u^2 - 2*u^3
    else return 1.0
    end
  end
end

"Return a constant function which is always evaluated to 1."
function constant()
  t::Float64 -> 1.0
end

end
