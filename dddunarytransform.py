import numpy as np
from scipy.sparse import diags

# we have ddd/one-hot interval representation
dim = 5
xd = np.array([0,0,1,0,0])
print("xd")
print(xd)

# The unary representation
D2U = 1.0 * np.fromfunction(lambda i,j: i <= j, (dim,dim))
print("D2U")
print(D2U)

# we have a unary representation
xu = D2U @ xd
print("xu")
print(xu)
xu_should_be = np.array([1, 1, 1, 0, 0])
assert((xu_should_be == xu).all())

U2D = diags([1,-1],[0,1], shape=(dim,dim)).toarray()
print("U2D")
print(U2D)

print("U2D*D2U")
print(U2D @ D2U)
print("D2U*U2D")
print(D2U @ U2D)

assert(((U2D @ D2U) == np.eye(dim)).all())
assert(((D2U @ U2D) == np.eye(dim)).all())


A = np.array(
        [
            [1,1,1,1,1],
            [-1,-1,-1,-1,-1],
            [1,0,0,0,0],
            [-1,0,0,0,0],
            [0,1,0,0,0],
            [0,-1,0,0,0],
            [0,0,1,0,0],
            [0,0,-1,0,0],
            [0,0,0,1,0],
            [0,0,0,-1,0],
            [0,0,0,0,1],
            [0,0,0,0,-1],
])

b = np.array([1,-1,1,0,1,0,1,0,1,0,1,0])
print(f"A\n", A)
print(b)
print(f"A*U2D\n", (A @ U2D))
print(b)


