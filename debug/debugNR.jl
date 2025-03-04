using BenchmarkTools

 

function update!(a,aold)
    a=a+[5.0,6.0,7.0]
    if a[1] % 2 != 0
        a=aold
    else
        aold=a
    end
    return a, aold
end
    
 

@benchmark begin
    a=[5.0,6.0,7.0]
aold=copy(a)
 for i in 1:10000
    a, aold= update!(a,aold)
end
end

 
 