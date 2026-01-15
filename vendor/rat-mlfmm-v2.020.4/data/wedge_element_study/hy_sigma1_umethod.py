import sympy as sp
from sympy.printing.pycode import pycode

# ============================================================================
# 1. Define symbols and parameters.
# ============================================================================
a, b, c, du, sigma1, mu0 = sp.symbols('a b c du sigma1 mu0', positive=True, real=True)
pi = sp.pi

# ============================================================================
# 2. Manually define triangle vertices (using the given geometry).
# ============================================================================
# Fixed vertices:
# O = sp.Matrix([0, 0, 0])
# A = sp.Matrix([a, 0, 0])
# B = sp.Matrix([a, b, 0])
# C = sp.Matrix([0, 0, c])
# # Shifted vertices (using lateral shift du):
# E = sp.Matrix([du*a/Dab, du*b/Dab, 0])
# F = sp.Matrix([a, du*b/Dab, 0])
# P = sp.Matrix([du*a/Dab, 0, 0])  # Projection point (given)

# ============================================================================
# 3. Manually express triangle side lengths in terms of a, b, du.
# Note that we assume a, b > 0 so that we can drop abs().
# ============================================================================
# Triangle EFB:
a1 = a - du*a/sp.sqrt(a**2 + b**2)
b1 = b - du*b/sp.sqrt(a**2 + b**2)

# Triangle AFE:
a2 = a - du*a/sp.sqrt(a**2 + b**2)    # horizontal side (same as a1)
b2 = du*b/sp.sqrt(a**2 + b**2)        # vertical side

# Triangle AEP:
a3 = du*b/sp.sqrt(a**2 + b**2)        # vertical difference between E and P
b3 = a - du*a/sp.sqrt(a**2 + b**2)    # horizontal difference between A and P

# Triangle OPE:
a4 = du*b/sp.sqrt(a**2 + b**2)        # same as a3
b4 = du*a/sp.sqrt(a**2 + b**2)        # from O to P

# ============================================================================
# 4. Define the potential functions F0, F1, and F2.
#    (We assume that all arguments are positive so that abs(cc)=cc.)
# ============================================================================
# Additional generic symbols used in the F-functions.
aa, bb, cc = sp.symbols('aa bb cc', real=True, positive=True)

# 2. Define the F-functions as given.
F0_expr = 1/(4*pi*mu0)*(
    (aa/2)*sp.log((sp.sqrt(aa**2+bb**2+cc**2)+bb)/(sp.sqrt(aa**2+bb**2+cc**2)-bb))
    - cc*sp.atan((aa*bb)/((aa**2+cc**2) + cc*sp.sqrt(aa**2+bb**2+cc**2)))
)
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
# 5. Form the net potential change (φ₂ - φ₁) as a sum of 9 terms.
# ============================================================================
phi1 = sigma1 * F1(a, b, c)

T1 = sigma1 * (du*a/sp.sqrt(a**2 + b**2)) * F0(a1, b1, c)
T2 = sigma1 * (du*a/sp.sqrt(a**2 + b**2)) * F0(a2, b2, c)
T3 = sigma1 * (du*a/sp.sqrt(a**2 + b**2)) * F0(a3, b3, c)
T4 = sigma1 * (du*a/sp.sqrt(a**2 + b**2)) * F0(a4, b4, c)
T5 = sigma1 * F1(a1, b1, c)
T6 = sigma1 * F1(a2, b2, c)
T7 = sigma1 * F2(a3, b3, c)
T8 = - sigma1 * F2(a4, b4, c)
T9 = - sigma1 * F1(a, b, c)

delta_phi = T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9


# 6. Technique 1: Compute the limit term-by-term as dx -> 0.
# L1 = sp.limit(sp.simplify(T1), du, 0)
# print(pycode(L1))
# L2 = sp.limit(sp.simplify(T2), du, 0)
# print(pycode(L2))
# L3 = sp.limit(sp.simplify(T3), du, 0)
# print(pycode(L3))
# L4 = sp.limit(sp.simplify(T4), du, 0)
# print(pycode(L4))
# L5 = sp.simplify(sp.limit(sp.simplify(T5), du, 0))
# print(pycode(L5))
# L6 = sp.simplify(sp.limit(sp.simplify(T6), du, 0))
# print(pycode(L6))
# L7 = sp.limit(sp.simplify(T7), du, 0)
# print(pycode(L7))
# L8 = sp.limit(sp.simplify(T8), du, 0)
# print(pycode(L8))
# L9 = sp.simplify(sp.limit(sp.simplify(T9), du, 0))
# print(pycode(L9))

# delta_phi_lin = L1 + L2 + L3 + L4 + L5 + L6 + L7 + L8 + L9


# ============================================================================
# 6. Extract the linear (du) term using series expansion.
# # ============================================================================
def linear_term(expr):
    return sp.series(expr, du, 0, 2).removeO().coeff(du)

L1 = linear_term(sp.expand(T1))
print(pycode(L1))
L2 = linear_term(sp.expand(T2))
print(pycode(L2))
L3 = linear_term(sp.expand(T3))
print(pycode(L3))
L4 = linear_term(sp.expand(T4))
print(pycode(L4))
L5 = linear_term(sp.expand(T5))
print(pycode(L5))
L6 = linear_term(sp.expand(T6))
print(pycode(L6))
L7 = linear_term(sp.expand(T7))
print(pycode(L7))
L8 = linear_term(sp.expand(T8))
print(pycode(L8))
L9 = linear_term(sp.expand(T9))
print(pycode(L9))

delta_phi_lin = sp.together(L1 + L2 + L3 + L4 + L5 + L6 + L7 + L8 + L9)

# delta_phi_lin = linear_term(sp.together(T1+T2+T3+T4+T5+T6+T7+T8+T9))


# According to the finite difference approximation, we set:
dphidu_lin = - delta_phi_lin   # minus sign is by convention

# ============================================================================
# 7. Use the known expression for dφ/dx.
# ============================================================================
dphidx = sp.simplify((sigma1/(4*pi*mu0)) * (
    c*sp.atan(a*b/(sp.sqrt(a**2 + c**2)**2 + c*sp.sqrt(a**2 + b**2 + c**2)))
    + (a*b/(sp.sqrt(a**2 + b**2 + c**2)*(sp.sqrt(a**2 + b**2 + c**2)-b)))*((b+c-sp.sqrt(a**2 + b**2 + c**2)) + (c*(b+c)*(c-sp.sqrt(a**2 + b**2 + c**2)))/(sp.sqrt(a**2 + b**2)**2))
))

# ============================================================================
# 8. Form the σ₁ contribution to Hy:
#     Hy_sigma1 = (Dab*d(φ)/du - a*d(φ)/dx)/b.
# ============================================================================
Hy_sigma1 = (sp.sqrt(a**2 + b**2)*dphidu_lin - a*dphidx) / b
Hy_sigma1_simpl = sp.simplify(Hy_sigma1)

# ============================================================================
# 9. Display the final symbolic expression and generate Python code.
# ============================================================================
print("The final symbolic expression for Hy_sigma1 (9-term split) is:")
sp.pprint(Hy_sigma1_simpl)
print("\nAs Python code:")
print(pycode(Hy_sigma1_simpl))


