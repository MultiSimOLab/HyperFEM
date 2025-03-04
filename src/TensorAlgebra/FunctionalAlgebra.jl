using Gridap

abstract type Geometry end

struct Box{T} <: Geometry
    ε::VectorValue{T,Float64}
    xc::VectorValue{T,Float64}
    function Box(xc::Vector{Float64},ε::Vector{Float64})
      @assert(length(xc) == length(ε) , "Inconsistent size of inputs")
      T=length(xc)
      new{T}(VectorValue(ε),VectorValue(xc))
    end

    function (obj::Box)(x::VectorValue{T,Float64}) where T
        out = 1.0
        for i in 1:T
            if x[i]  < obj.xc[i]-obj.ε[i] || x[i] > obj.xc[i]+obj.ε[i]
                out *=0.0
            end
        end
        return out
    end



    function (obj::Box)(x::Vector{Float64})
         obj(VectorValue(x))
    end

    function (obj::Box)(func::Function)
        return (x)-> obj(x)*func(x)
   end
end



struct Ellipsoid{T} <: Geometry
    ε::VectorValue{T,Float64}
    xc::VectorValue{T,Float64}
    function Ellipsoid(xc::Vector{Float64},ε::Vector{Float64})
      @assert(length(xc) == length(ε) , "Inconsistent size of inputs")
      T=length(xc)
      new{T}(VectorValue(ε),VectorValue(xc))
    end


    function (obj::Ellipsoid)(x::VectorValue{T,Float64}) where T
        diff = (x .- obj.xc) ./ obj.ε
        value = sum(diff.data .^ 2)
        return value <= 1.0 ? 1.0 : 0.0
    end


    function (obj::Ellipsoid)(x::Vector{Float64})
         obj(VectorValue(x))
    end
 
    function (obj::Ellipsoid)(func::Function)
        return (x)-> obj(x)*func(x)
   end

end



# xc=[0.0,0.0,0.0]
# ε=[1.0,1.0,1.0]
# #creo objeto de tipo Box
# a=Box(xc,ε)
# b=Ellipsoid(xc,ε)

# # filtro funcion f y evalue en punto
# f(x)=x[1]+x[2]+x[3]
# @time b([1.0,1.0,0.9])

# x=VectorValue([1.0,1.0,0.9])
# xc=VectorValue([1.0,1.0,0.9])
# ε=VectorValue([1.0,1.0,0.9])

# diff = (x .- xc) ./ ε
# value = sum(diff.^ 2)
