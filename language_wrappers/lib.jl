module NMFMergeLib

using NMFMerge

struct ReturnValue
    niters::Int32
    converged::Bool
    objvalue::Float64
end

Base.@ccallable function nmfmerge_inplace(Wout::Matrix{Float64}, Hout::Matrix{Float64}, X::Matrix{Float64}, ncomponents::Int32, tol::Float64, maxiter::Int32)::ReturnValue
    result = nmfmerge(X, ncomponents; tol_final=tol, maxiter)
    Wout .= result.W
    Hout .= result.H
    return ReturnValue(result.niters, result.converged, result.objvalue)
end

end
