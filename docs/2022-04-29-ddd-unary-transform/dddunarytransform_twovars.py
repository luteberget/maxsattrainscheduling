import numpy as np
from scipy.sparse import diags


def convert_onehot(dim):
    return 1.0 * np.fromfunction(lambda i, j: i <= j, (dim, dim))


def convert_unary(dim):
    return diags([1, -1], [0, 1], shape=(dim, dim)).toarray()


def t(dim, convert):
    d = convert(dim)
    z = np.zeros((dim, dim))
    return np.block([[d, z], [z, d]])


def number(dim, x):
    z = np.zeros((dim,))
    if x < dim:
        z[x] = 1
    return z


def t1plus1gtt2_onehot(dim):
    A = np.vstack([np.concatenate([number(dim, x), number(dim, y)])
                   for x in range(dim)
                   for y in range(dim)
                   if y <= x])
    b = np.ones((dim,))
    return (A, b)


def t1plus1gtt2_unary(dim):
    A = np.vstack([np.concatenate([number(dim, x), -1.0 * number(dim, x+1)])
                   for x in range(dim)])
    b = np.zeros((dim,))
    return (A, b)


A, b = t1plus1gtt2_unary(3)
print(A)
print(b)

print (A @ t(3, convert_onehot))


# # we have ddd/one-hot interval representation
# dim = 5
# xd = np.array([0,0,1,0,0])
# print("xd")
# print(xd)

# # The unary representation
# D2U = 1.0 * np.fromfunction(lambda i,j: i <= j, (dim,dim))
# print("D2U")
# print(D2U)

# # we have a unary representation
# xu = D2U @ xd
# print("xu")
# print(xu)
# xu_should_be = np.array([1, 1, 1, 0, 0])
# assert((xu_should_be == xu).all())

# U2D = diags([1,-1],[0,1], shape=(dim,dim)).toarray()
# print("U2D")
# print(U2D)

# print("U2D*D2U")
# print(U2D @ D2U)
# print("D2U*U2D")
# print(D2U @ U2D)

# assert(((U2D @ D2U) == np.eye(dim)).all())
# assert(((D2U @ U2D) == np.eye(dim)).all())


# A = np.array(
#         [
#             [1,1,1,1,1],
#             [-1,-1,-1,-1,-1],
#             [1,0,0,0,0],
#             [-1,0,0,0,0],
#             [0,1,0,0,0],
#             [0,-1,0,0,0],
#             [0,0,1,0,0],
#             [0,0,-1,0,0],
#             [0,0,0,1,0],
#             [0,0,0,-1,0],
#             [0,0,0,0,1],
#             [0,0,0,0,-1],
# ])

# b = np.array([1,-1,1,0,1,0,1,0,1,0,1,0])
# print(f"A\n", A)
# print(b)
# print(f"A*U2D\n", (A @ U2D))
# print(b)
