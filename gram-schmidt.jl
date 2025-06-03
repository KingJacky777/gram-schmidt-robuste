using LinearAlgebra

function gramschmidt(A::Matrix)
    Q = zeros(eltype(A), size(A))

    for i in axes(A, 2)
        Q[:, i] = A[:, i]

        for j in 1:i-1
            Q[:, i] -= dot(Q[:, j], Q[:, i]) * Q[:, j]
        end

        Q[:, i] = Q[:, i] / norm(Q[:, i])
    end

    return Q
end
