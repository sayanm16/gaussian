import streamlit as st
import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    # Forward elimination
    for i in range(n):
        # Make the diagonal element 1
        diag_element = A[i][i]
        A[i] = A[i] / diag_element
        b[i] = b[i] / diag_element

        # Make the elements below the diagonal 0
        for j in range(i + 1, n):
            factor = A[j][i]
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i] - np.dot(A[i][i + 1:], x[i + 1:])

    return x

st.title("Linear Equation Solver using Gaussian Elimination")

num_equations = st.number_input("Number of equations", min_value=1, max_value=10, value=3, step=1)

equations = []
for i in range(int(num_equations)):
    eq = st.text_input(f"Equation {i+1}", value="0 0 0 0", help="Enter coefficients and constant separated by spaces")
    equations.append(eq)

if st.button("Solve"):
    try:
        A = []
        b = []
        for eq in equations:
            coeffs = list(map(float, eq.split()))
            A.append(coeffs[:-1])
            b.append(coeffs[-1])
        
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        
        solution = gaussian_elimination(A, b)
        
        st.subheader("Solution:")
        for i, val in enumerate(solution):
            st.write(f"x{i+1} = {val:.6f}")
        
        # Verify the solution
        st.subheader("Verification:")
        for i, eq in enumerate(equations):
            coeffs = list(map(float, eq.split()))
            result = sum(c * x for c, x in zip(coeffs[:-1], solution))
            st.write(f"Equation {i+1}: {result:.6f} ≈ {coeffs[-1]}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.sidebar.markdown("""
## How to use:
1. Enter the number of equations.
2. For each equation, enter the coefficients and constant separated by spaces.
   For example, for the equation 2x + 3y - z = 5, enter: 2 3 -1 5
3. Click the "Solve" button to see the solution.
""")
