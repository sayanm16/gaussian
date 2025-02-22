import streamlit as st
import numpy as np

def gaussian_elimination_pivoting(A, b):
    """
    Solves the system of linear equations Ax = b using Gaussian elimination with partial pivoting.
    """
    n = len(b)

    for i in range(n):
        # Step 1: Pivoting (Swap rows if needed)
        max_row = i + np.argmax(np.abs(A[i:, i]))  # Find the row with the largest value in column i
        if i != max_row:
            A[[i, max_row]] = A[[max_row, i]]  # Swap rows in A
            b[i], b[max_row] = b[max_row], b[i]  # Swap elements in b
        
        # Step 2: Forward Elimination
        diag_element = A[i][i]
        if abs(diag_element) < 1e-12:  # Check for near-zero pivot
            st.error("Singular or nearly singular matrix detected. Cannot proceed.")
            return None

        A[i] = A[i] / diag_element  # Normalize row
        b[i] = b[i] / diag_element

        for j in range(i + 1, n):
            factor = A[j][i]
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]

    # Step 3: Back Substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i] - np.dot(A[i][i + 1:], x[i + 1:])

    return x

# Streamlit UI
st.title("Linear Equation Solver using Gaussian Elimination with Pivoting")

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
            if len(coeffs) != num_equations + 1:
                st.error(f"Equation {eq} does not have the correct number of terms. Please check your input.")
                st.stop()

            A.append(coeffs[:-1])
            b.append(coeffs[-1])
        
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        
        solution = gaussian_elimination_pivoting(A, b)
        
        if solution is not None:
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
   Example: For `2x + 3y - z = 5`, enter: `2 3 -1 5`
3. Click "Solve" to compute the solution.
4. The system verifies the computed values.
""")
