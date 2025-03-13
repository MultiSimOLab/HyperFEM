import Base.min
import Base.max
 

#*******************************************************************************	
#                 OptimVariables
#*******************************************************************************	



abstract type OptimVariable end


struct OptimFEVariable{A,B,C} <: OptimVariable
  func::A # FEFunction  
  filter::B
  bounds::Vector{Float64}
  caches::C
  function OptimFEVariable(func::FEFunction, bounds::Vector{Float64}=[0.0,1.0], filter=(x)->x)
    func.free_values .=  map((x) -> max(bounds[1], min(x, bounds[2])), func.free_values)
    xold = deepcopy(get_free_dof_values(func))
    caches  = (xold,)
    A, B, C = typeof(func), typeof(filter), typeof(caches)
    new{A,B,C}(func, filter, bounds, caches)
  end
end

function update!(obj::OptimFEVariable, vec::Vector)  
  # update old
  obj.caches[1] .= obj.func.free_values
  # filter new
  filter(vec)
  # update new
  obj.func.free_values .=  map((x) -> max(obj.bounds[1], min(x, obj.bounds[2])), vec)
end

function pushforward!(obj::OptimFEVariable, vec::Vector)  
  # update old
  obj.caches[1] .= obj.func.free_values
  # filter new
  filter(vec)
  # update new
  obj.func.free_values .+= vec
  # apply bounds
  obj.func.free_values .= map((x) -> max(obj.bounds[1], min(x, obj.bounds[2])), obj.func.free_values)
end

function pullback!(obj::OptimFEVariable)  
  # update new
  obj.func.free_values .= obj.caches[1]
end
 
 
get_state(obj::OptimFEVariable) = obj.func




#*******************************************************************************	
#                 FEFunctional
#*******************************************************************************	

struct FEFunctional{A}
  J::Function
  DJ::Function
  caches::A
  function FEFunctional(J::Function, DJ::Function, uh, ph, ϕh)
    Vϕ = ϕh.fe_space
    dj = assemble_vector(DJ(uh, ph, ϕh), Vϕ)
    Jadim = [1.0]
    caches = (uh, ph, ϕh, Vϕ, dj, Jadim)
    # falta meter en cache vector x de derivadas
    A = typeof(caches)
    new{A}(J, DJ, caches)
  end
end

function adimensionalize!(func::FEFunctional, Jadim)
  func.caches[6][1] = Jadim
  func.caches[5] ./= Jadim
  return 1.0, func.caches[5]
end

function evaluate_objective(func::FEFunctional)
  uh, _, ϕh, _, _, Jadim = func.caches
  jadim = Jadim[1]
  sum(func.J(uh, ϕh)) / jadim
end

function evaluate_derivative!(func::FEFunctional)
  uh, ph, ϕh, Vϕ, dj, Jadim = func.caches
  jadim = Jadim[1]
  assemble_vector!(func.DJ(uh, ph, ϕh), dj, Vϕ)
  dj ./= jadim
  return dj
end

function evaluate!(func::FEFunctional)
  j = evaluate_objective(func::FEFunctional)
  dj = evaluate_derivative!(func::FEFunctional)
  return j, dj
end

get_space(func::FEFunctional) = func.caches[4]
get_derivative(func::FEFunctional) = FEFunction(get_space(func), func.caches[5])
