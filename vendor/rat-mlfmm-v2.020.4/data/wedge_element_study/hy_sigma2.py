import sympy as sp
from sympy.printing.pycode import pycode

# 1. Define symbols (all assumed real and positive)
a, b, c, dy, mu0, sigma2 = sp.symbols('a b c dy mu0 sigma2', real=True, positive=True)
pi = sp.pi

# Additional generic symbols used in the F-functions.
aa, bb, cc = sp.symbols('aa bb cc', real=True, positive=True)

# 2. Define the F-functions exactly as in your MATLAB code.
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

F0 = sp.Lambda((aa, bb, cc), sp.simplify(F0_expr))
F1 = sp.Lambda((aa, bb, cc), sp.simplify(F1_expr))
F2 = sp.Lambda((aa, bb, cc), sp.simplify(F2_expr))

# 3. Define the points of the source triangle in R^3.
# Following MATLAB:
# O = [0, 0, 0], A = [a, 0, 0], B = [a, b, 0], E = [0, dy, 0]
O_vec = sp.Matrix([0, 0, 0])
A_vec = sp.Matrix([a, 0, 0])
B_vec = sp.Matrix([a, b, 0])
E_vec = sp.Matrix([0, dy, 0])

# Also define F_vec as the projection of E onto AB.
# In MATLAB, F = [a, dy, 0]
F_vec = sp.Matrix([a, dy, 0])

# 4. Compute projection of E onto OB.
V = (B_vec - O_vec) / sp.sqrt(a**2 + b**2)  # V = (a, b, 0) normalized
t = (E_vec - O_vec).dot(V)                   # t = dot(E - O, V)
P = O_vec + t*V                            # Projection point on OB

# For Triangle EFB:
a1 = a                       # norm(F - E) = sqrt((a-0)^2+((dy-dy)^2)) = a.
# norm(F - B) = sqrt((a-a)^2 + (dy-b)^2) = |b - dy|
b1 = b - dy

# For Triangle AFE:
a2 = a                       # same as F - E = a.
b2 = dy                      # norm(F - A) = dy.

# For Triangle AEO:
a3 = dy                      # norm(E - O) = dy.
b3 = a                       # norm(A - O) = a.

# For Triangle OPE:
# We have E = (0, dy, 0) and we compute the projection P of E on OB.
# OB runs from O=(0,0,0) to B=(a, b, 0), so its length is sqrt(a^2+b^2).
# t = dot(E-O, OB)/|OB| = (dy * b)/sqrt(a^2+b^2).
# So P = (t*a/sqrt(a^2+b^2), t*b/sqrt(a^2+b^2), 0) = (dy*a*b/(a^2+b^2), dy*b^2/(a^2+b^2), 0).
P_x = dy * a * b/(a**2+b**2)
P_y = dy * b**2/(a**2+b**2)
# Then, OPE:
a4 = dy * a/sp.sqrt(a**2+b**2)  # norm(E - P) simplifies to dy*a/sqrt(a^2+b^2)
b4 = dy * b/sp.sqrt(a**2+b**2)  # norm(P - O) = dy*b/sqrt(a^2+b**2)

# For Triangle PBE:
a5 = a4  # same as norm(E-P)
# norm(P - B): P = (dy*a*b/(a**2+b**2), dy*b**2/(a**2+b**2), 0) and B = (a, b, 0)
b5 = sp.sqrt((P_x - a)**2 + (P_y - b)**2)

# 5. Express the geometric factors sigmaEP and sigmaOB in closed–form.
# From the MATLAB code these are known (from our previous derivation):
sigmaEP = sigma2 * a/sp.sqrt(a**2+b**2)
sigmaOB = - sigma2 * b/sp.sqrt(a**2+b**2)

# 7. Define the potential terms.
# phi1 (unshifted potential) for sigma2:
phi1 = sigma2 * F2(a, b, c)

# Now define 11 separate terms analogous to the MATLAB code.
T1  = sigma2 * dy * F0(a1, b1, c)
T2  = sigma2 * dy * F0(a2, b2, c)
T3  = sigma2 * dy * F0(a3, b3, c)
T4  = sigma2 * dy * F0(a4, b4, c)
T5  = sigma2 * dy * F0(a5, b5, c)
T6  = sigma2           * F2(a1, b1, c)
T7  = sigma2           * F2(a2, b2, c)
T8  = sigma2           * F1(a3, b3, c)
T9  = - sigmaEP        * F1(a4, b4, c)
T10 = - sigmaEP        * F1(a5, b5, c)
T11 = - sigmaOB        * F2(a4, b4, c) 
T12 = sigmaOB        * F2(a5, b5, c)

phi2 = T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11 + T12

# 8. Define a helper function to extract the coefficient of dy (the linear term).
def linear_term(expr, var):
    # Expand expr as a series in var to order 2, remove the O() term, and get the coefficient of var.
    return sp.series(expr, var, 0, 2).removeO().coeff(var, 1)

# # Compute the linear term for each T_i separately.
# L1 = linear_term(sp.expand(T1), dy)
# print(pycode(L1))
# L2 = linear_term(sp.expand(T2), dy)
# print(pycode(L2))
# L3 = linear_term(sp.expand(T3), dy)
# print(pycode(L3))
# L4 = linear_term(sp.expand(T4), dy)
# print(pycode(L4))
# L5 = linear_term(sp.expand(T5), dy)
# print(pycode(L5))
# L6 = linear_term(sp.expand(T6), dy)
# print(pycode(L6))
# L7 = linear_term(sp.expand(T7), dy)
# print(pycode(L7))
# L8 = linear_term(sp.expand(T8), dy)
# print(pycode(L8))
# L9 = linear_term(sp.expand(T9), dy)
# print(pycode(L9))
# L10 = linear_term(sp.expand(T10), dy)
# print(pycode(L10))
# L11 = linear_term(sp.expand(T11), dy)
# print(pycode(L11))
# L12 = linear_term(sp.expand(T12), dy)
# print(pycode(L12))

# # Sum the individual linear terms.
# phi_lim_series = L1 + L2 + L3 + L4 + L5 + L6 + L7 + L8 + L9 + L10 + L11 + L12

# Compute the linear term for each T_i separately.
phi_lim_series = linear_term(sp.expand(phi2), dy)


# (Note: phi1 is independent of dy, so its derivative is zero.)

# 9. The finite-difference derivative gives the field:
Hy_sigma2_lim = - sp.simplify(phi_lim_series)

# 10. Optionally, simplify further.
Hy_sigma2_lim = sp.factor(Hy_sigma2_lim)
Hy_sigma2_lim = sp.trigsimp(Hy_sigma2_lim)
Hy_sigma2_lim = sp.simplify(Hy_sigma2_lim)

# 11. Display the final expression (in Python code format).
code_str = pycode(Hy_sigma2_lim)
print("Hy_sigma2_lim = " + code_str)
