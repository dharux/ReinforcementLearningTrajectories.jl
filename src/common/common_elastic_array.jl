using ElasticArrays: ElasticArray, resize_lastdim!

#####
# extensions for ElasticArrays
#####

Base.push!(a::ElasticArray, x) = append!(a, x)
Base.push!(a::ElasticArray{T,1}, x) where {T} = append!(a, [x])
Base.empty!(a::ElasticArray) = resize_lastdim!(a, 0)
