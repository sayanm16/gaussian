import tkinter as tk
from tkinter import messagebox
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

def solve_equations():
    try:
        # Get the number of equations
        n = int(num_equations.get())
        
        # Get the coefficients and constants
        A = []
        b = []
        for i in range(n):
            eq = equation_entries[i].get().split()
            A.append([float(x) for x in eq[:-1]])
            b.append(float(eq[-1]))
        
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        
        # Solve the system
        solution = gaussian_elimination(A, b)
        
        # Display the solution
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Solution:\n")
        for i, val in enumerate(solution):
            result_text.insert(tk.END, f"x{i+1} = {val:.6f}\n")
    
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main window
root = tk.Tk()
root.title("Linear Equation Solver")

# Create and pack the input widgets
tk.Label(root, text="Number of equations:").pack()
num_equations = tk.Entry(root)
num_equations.pack()

tk.Button(root, text="Set", command=lambda: create_equation_inputs()).pack()

equation_frame = tk.Frame(root)
equation_frame.pack()

equation_entries = []

def create_equation_inputs():
    # Clear previous entries
    for widget in equation_frame.winfo_children():
        widget.destroy()
    equation_entries.clear()
    
    try:
        n = int(num_equations.get())
        for i in range(n):
            tk.Label(equation_frame, text=f"Equation {i+1}:").pack()
            entry = tk.Entry(equation_frame, width=50)
            entry.pack()
            equation_entries.append(entry)
        
        tk.Button(equation_frame, text="Solve", command=solve_equations).pack()
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid number of equations.")

# Create and pack the output widget
result_text = tk.Text(root, height=10, width=50)
result_text.pack()

root.mainloop()
