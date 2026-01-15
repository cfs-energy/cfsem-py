import sympy as sp
from sympy.printing.pycode import pycode

# Define symbols (we assume a, b, c, sigma2, mu0 > 0 so that square roots and logs are unambiguous)
a, b, c, sigma2, mu0 = sp.symbols('a b c sigma2 mu0', positive=True, real=True)

# Define the auxiliary subexpressions
Dabc = sp.sqrt(a**2 + b**2 + c**2)  # Full 3D distance
Dab = sp.sqrt(a**2 + b**2)           # In-plane distance (from a and b)
Dac = sp.sqrt(a**2 + c**2)           # (You call this Dac so that Dac^2 = a^2+c^2)

# --- Original lengthy expression ---
# We use L, T, R as in earlier steps.
L = sp.log(-1/(b - Dabc))
T = sp.atan(a*b/(a**2 + c**2 + c*Dabc))
R = sp.sqrt((a**2 + b**2)*(a**2 + b**2 + c**2))

expr_num = ( a**5 * L 
           - 2*a**4*c * T
           + 2*a**3*b**2 * L 
           + 2*a**3*b*c
           + 2*a**3*b*sp.sqrt(a**2 + b**2)
           - 2*a**3*b*Dabc
           - a**3*R * L 
           - 4*a**2*b**2*c * T 
           + 2*a**2*c*R * T
           + a*b**4 * L
           + 2*a*b**3*c
           + 2*a*b**3*sp.sqrt(a**2 + b**2)
           - 2*a*b**3*Dabc
           - a*b**2*R * L
           + 2*a*b*c**2*sp.sqrt(a**2 + b**2)
           - 2*a*b*c*R
           - 2*b**4*c * T
           + 2*b**2*c*R * T
           + sp.log((b + Dabc)**(a*(a**4+2*a**2*b**2 - a**2*sp.sqrt(a**4+2*a**2*b**2+a**2*c**2+b**4+b**2*c**2)
                       + b**4-b**2*sp.sqrt(a**4+2*a**2*b**2+a**2*c**2+b**4+b**2*c**2))))
           )
denom = sp.pi*mu0*(a**2+b**2)*(a**2+b**2 - sp.sqrt(a**2+b**2)*Dabc)
original_expr = -sigma2/(8) * expr_num / denom

# --- Candidate simplified expression ---
# Your candidate is written in terms of Dabc, Dab, Dac.
candidate_expr = (sigma2/(4*sp.pi*mu0)) * (
                   c*sp.atan(a*b/(Dac**2 + c*Dabc)) +
                   (a*b/(Dabc*(Dabc - Dab))) * (
                      (Dab + c - Dabc) + (c*(Dab + c)*(c - Dabc))/(Dab**2)
                   )
                 )

# --- Compare the two expressions ---
# We first attempt to simplify their difference symbolically.
diff_expr = sp.simplify(original_expr - candidate_expr)
print("Symbolic difference:")
print(pycode(diff_expr))
print("\n")

# --- Test numerical values ---
test_vals = {a: 1, b: 2, c: 3, sigma2: 1, mu0: 1}
orig_val = original_expr.evalf(subs=test_vals)
cand_val = candidate_expr.evalf(subs=test_vals)
diff_val = diff_expr.evalf(subs=test_vals)

print("Original expression numerical value:", orig_val)
print("Candidate expression numerical value:", cand_val)
print("Difference:", diff_val)