import numpy as np

def generate_matrix(size):
    """Generate a random non-singular matrix."""
    matrix = np.random.randint(0, 2, size)
    while np.linalg.matrix_rank(matrix) < size[0]:
        matrix = np.random.randint(0, 2, size)
    return matrix

def generate_permutation_matrix(size):
    """Generate a random permutation matrix."""
    perm = np.random.permutation(size)
    matrix = np.zeros((size, size))
    for i in range(size):
        matrix[i, perm[i]] = 1
    return matrix

def create_goppa_and_ldpc_keys(k, n, m, t, delta, l, a):
    """Generate keys for the LDPC-based McEliece cryptosystem."""
    # Generate Goppa parameters
    S = generate_matrix((k, k))
    P = generate_permutation_matrix(n)
    G = generate_matrix((k, n))  # Goppa code generator matrix

    U1 = np.dot(S, np.dot(G, P))  # Inner U1 matrix

    # Expand U1 matrix for multiple blocks
    U1_full = np.zeros((k * l, n * l))
    for i in range(l):
        U1_full[i * k:(i + 1) * k, i * n:(i + 1) * n] = U1

    # Generate LDPC parameters
    S_hat = generate_matrix((n * l, n * l))
    P_hat = generate_permutation_matrix(m)
    G_hat = generate_matrix((n * l, m))  # LDPC generator matrix

    U2 = np.dot(S_hat, np.dot(G_hat, P_hat))  # Outer U2 matrix

    # Final public key matrix
    U = np.dot(U1_full, U2)

    # Generate the list A
    canonical_base = [np.eye(n)[i] for i in range(n)]
    A = []
    for _ in range(a):
        selected_vectors = np.random.choice(canonical_base, l, replace=False)
        concatenated_vectors = np.concatenate(selected_vectors)
        expanded_vector = np.dot(concatenated_vectors, U2)
        A.append(expanded_vector)

    # Public and private keys
    public_key = (U, A, t, delta)
    private_key = (S, P, G, S_hat, P_hat, G_hat)

    return public_key, private_key

# Example parameters
k, n, m, t, delta, l, a = 10, 20, 30, 5, 3, 2, 50
public_key, private_key = create_goppa_and_ldpc_keys(k, n, m, t, delta, l, a)
print("Public Key:", public_key)
print("Private Key:", private_key)
