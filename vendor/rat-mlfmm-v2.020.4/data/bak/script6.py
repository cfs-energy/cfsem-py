import sympy as sp
from sympy.printing.pycode import pycode

# ============================================================================
# 1. Define symbols and parameters.
# ============================================================================
a, b, c, dy, sigma1, mu0 = sp.symbols('a b c dy sigma1 mu0', positive=True, real=True)
pi = sp.pi

# ============================================================================
# 2. Define points for the geometric factors.
#    (We need these only to compute the geometric dot–products.)
# ============================================================================
O = sp.Matrix([0, 0, 0])
A = sp.Matrix([a, 0, 0])
B = sp.Matrix([a, b, 0])
E = sp.Matrix([0, dy, 0])
# Projection: P is the projection of E onto OB.
OB = B - O
norm_OB = sp.sqrt(OB.dot(OB))
V = OB / norm_OB
t = (E - O).dot(V)
P = O + t * V

# Compute geometric factors:
#   sigmaEP = sigma1 * ((P-E)/||P-E|| · (A-O)/||A-O||)
#   sigmaOB = sigma1 * ((O-B)/||B-O|| · (A-O)/||A-O||)
sigmaEP = sigma1 * ((P - E) / sp.sqrt((P - E).dot(P - E))).dot((A - O) / sp.sqrt((A - O).dot(A - O)))
sigmaOB = sigma1 * ((O - B) / sp.sqrt((B - O).dot(B - O))).dot((A - O) / sp.sqrt((A - O).dot(A - O)))
sigmaEP = sp.simplify(sigmaEP)
sigmaOB = sp.simplify(sigmaOB)

# ============================================================================
# 3. Write triangle side lengths directly in terms of a, b, and dy.
# ============================================================================
# Triangle EFB:
a1_expr = a         # distance between F=[a,dy,0] and E=[0,dy,0]
b1_expr = b - dy    # vertical difference: F to B, with B=[a,b,0]

# Triangle AFE:
a2_expr = a         # distance F to E (same horizontal extent)
b2_expr = dy        # vertical difference F to A (A=[a,0,0])

# Triangle AEO:
a3_expr = dy        # difference between E=[0,dy,0] and O=[0,0,0]
b3_expr = a         # difference between A=[a,0,0] and O=[0,0,0]

# Triangle OPE:
a4_expr = a*dy/sp.sqrt(a**2 + b**2)   # from similarity in triangle OPE
b4_expr = b*dy/sp.sqrt(a**2 + b**2)

# Triangle PBE:
# Here we determine b5 from a series expansion:
# P-B = P - [a, b, 0] and when expanded in dy, its norm is approximately:
#   b5 = Dab*(1 - dy*(b/(a**2+b**2)))
a5_expr = a4_expr     # we keep the same horizontal projection
# b5_expr = Dab*(1 - dy*b/(a**2+b**2))
b5_expr = sp.sqrt(a**2 + b**2) 

# ============================================================================
# 4. Define the potential functions F1 and F2.
# ============================================================================
aa, bb, cc = sp.symbols('aa bb cc', positive=True, real=True)

# F0_expr = 1/(4*pi*mu0)*(
#     (aa/2)*sp.log((sp.sqrt(aa**2+bb**2+cc**2)+bb)/(sp.sqrt(aa**2+bb**2+cc**2)-bb))
#     - cc*sp.atan((aa*bb)/((aa**2+cc**2) + cc*sp.sqrt(aa**2+bb**2+cc**2)))
# )
F1_expr = 1/(4*pi*mu0)*(
    ((aa**2+cc**2)/4)*sp.log((sp.sqrt(aa**2+bb**2+cc**2)+bb)/(sp.sqrt(aa**2+bb**2+cc**2)-bb))
    - (bb*cc**2/(2*sp.sqrt(aa**2+bb**2)))*sp.log((sp.sqrt(aa**2+bb**2)+sp.sqrt(aa**2+bb**2+cc**2))/cc)
)
F2_expr = 1/(4*pi*mu0)*(
    (aa/2)*(sp.sqrt(aa**2+bb**2+cc**2)-sp.sqrt(aa**2+cc**2))
    + (aa*cc**2/(2*sp.sqrt(aa**2+bb**2)))*sp.log((sp.sqrt(aa**2+bb**2)+sp.sqrt(aa**2+bb**2+cc**2))/cc)
    - (cc**2/2)*sp.log((sp.sqrt(aa**2+cc**2)+aa)/cc)
)
# Simplify and create callable functions.
F0 = sp.Lambda((aa, bb, cc), sp.simplify(F0_expr))
F1 = sp.Lambda((aa, bb, cc), sp.simplify(F1_expr))
F2 = sp.Lambda((aa, bb, cc), sp.simplify(F2_expr))


# ============================================================================
# 5. Construct the unshifted and shifted potentials.
# ============================================================================
phi1 = sigma1 * F1(a, b, c)

# Shifted potential from extra regions:
phi2 = sigma1 * ( F1(a1_expr, b1_expr, c) + F1(a2_expr, b2_expr, c) + F2(a3_expr, b3_expr, c) ) \
       - ( sigmaEP*( F1(a4_expr, b4_expr, c) + F1(a5_expr, b5_expr, c) )
           + sigmaOB*( F2(a4_expr, b4_expr, c) - F2(a5_expr, b5_expr, c) ) )

# ============================================================================
# 6. Split the net potential change Δφ = φ₂ - φ₁ into terms.
# ============================================================================
T1 = sigma1 * F1(a1_expr, b1_expr, c)
T2 = sigma1 * F1(a2_expr, b2_expr, c)
T3 = sigma1 * F2(a3_expr, b3_expr, c)
T4 = - sigmaEP * F1(a4_expr, b4_expr, c)
T5 = - sigmaEP * F1(a5_expr, b5_expr, c)
T6 = - sigmaOB * F2(a4_expr, b4_expr, c)
T7 = sigmaOB * F2(a5_expr, b5_expr, c)
T8 = - sigma1 * F1(a, b, c)   # subtract unshifted potential

delta_phi = T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8

# ============================================================================
# 7. Extract the linear (dy) term via series expansion.
# ============================================================================
def linear_term(expr):
    # Expand expr in dy to order dy^2 and extract the coefficient of dy.
    return sp.series(expr, dy, 0, 2).removeO().coeff(dy)

series_T1 = linear_term(T1)
print(pycode(series_T1))
series_T2 = linear_term(T2)
print(pycode(series_T2))
series_T3 = linear_term(T3)
print(pycode(series_T3))
series_T4 = linear_term(T4)
print(pycode(series_T4))
series_T5 = linear_term(T5)
print(pycode(series_T5))
series_T6 = linear_term(T6)
print(pycode(series_T6))
series_T7 = linear_term(T7)
print(pycode(series_T7))
series_T8 = linear_term(T8)
print(pycode(series_T8))

phi_lim_series = sp.together(series_T1 + series_T2 + series_T3 + series_T4 + series_T5 + series_T6 + series_T7 + series_T8)

Hy_sigma1_lin = -dphi_dy_simpl

# ============================================================================
# 8. Substitute common radical expressions for clarity.
# ============================================================================
# try simplify further
# Hy_sigma1_improved = sp.factor(Hy_sigma1_lin)
# Hy_sigma1_improved = sp.trigsimp(Hy_sigma1_improved)
Hy_sigma1_improved = sp.simplify(Hy_sigma1_lin)

# ============================================================================
# 9. Output the final symbolic expression.
# ============================================================================
print("\nAs Python code:")
print(pycode(Hy_sigma1_improved))


