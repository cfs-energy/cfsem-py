import sympy as sp
from sympy.printing.pycode import pycode

# 1. Define symbols and assume positivity.
a, b, c, dx, mu0, sigma1 = sp.symbols('a b c dx mu0 sigma1', real=True, positive=True)
pi = sp.pi

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

# 3. Define the shifted variables.
# For the main shift:
a1 = a - dx
b1 = b

# For the two small triangles: 
a2 = (b/sp.sqrt(a**2 + b**2)) * dx
b2 = (a/sp.sqrt(a**2 + b**2)) * dx

# For the second triangle, a3 is like a2, while b3 remains constant to leading order.
a3 = a2
b3 = sp.sqrt(a**2 + b**2) # - b2

# 4. Geometric factors from the dot products.
sigmaEP = -sigma1*(b/sp.sqrt(a**2+b**2))
sigmaOB = -sigma1*(a/sp.sqrt(a**2+b**2))

# 5. Define the unshifted potential phi1 and the shifted potential phi2.
phi1 = sigma1 * F1(a, b, c)

# Instead of one grouped expression for phi2, we split into nine terms:
#  3 terms from sigma1*dx*F0
T1 = sigma1 * dx * F0(a1, b1, c)
T2 = sigma1 * dx * F0(a2, b2, c)
T3 = sigma1 * dx * F0(a3, b3, c)
#  2 terms from sigma1*F1 with the unshifted potential subtracted:
T4 = sigma1 * F1(a1, b1, c)
T5 = - sigma1 * F1(a, b, c)  # subtracting the original phi1 part.
#  2 terms from sigmaEP*F1:
T6 = sigmaEP * F1(a2, b2, c)
T7 = sigmaEP * F1(a3, b3, c)
#  2 terms from sigmaOB*F2:
T8 = sigmaOB * F2(a2, b2, c)
T9 = -sigmaOB * F2(a3, b3, c)

# The combined shifted potential difference is:
#phi2_minus_phi1 = T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9
# phi2_minus_phi1 = T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9


#6. Technique 1: Compute the limit term-by-term as dx -> 0.
# lim_T1 = sp.limit(T1, dx, 0) # yes
# lim_T2 = sp.limit(T2, dx, 0) # no
# lim_T3 = sp.limit(T3, dx, 0) # no
# lim_T4 = sp.limit(T4, dx, 0) # yes big term
# lim_T5 = sp.limit(T5, dx, 0) # no
# lim_T6 = sp.limit(T6, dx, 0) # no
# lim_T7 = sp.limit(T7, dx, 0) # no
# lim_T8 = sp.limit(T8, dx, 0) # no
# lim_T9 = sp.limit(T9, dx, 0) # yes

# phi_lim_limit = sp.together(lim_T1 + lim_T2 + lim_T3 + lim_T4 + lim_T5 + lim_T6 + lim_T7 + lim_T8 + lim_T9)


# # 7. Technique 2: Use series expansion to extract the linear (dx) term.
# Define a helper to return the coefficient of dx from the series.
def linear_term(expr):
    # Expand to order dx^2 and extract the dx term.
    return sp.series(expr, dx, 0, 2).removeO().coeff(dx)

series_T1 = linear_term(T1)
print(pycode(series_T1))
# series_T2 = linear_term(T2)
# series_T3 = linear_term(T3)
series_T4 = linear_term(T4)
print(pycode(series_T4))
# series_T5 = linear_term(T5)
# series_T6 = linear_term(T6)
# series_T7 = linear_term(T7)
# series_T8 = linear_term(T8)
series_T9 = linear_term(T9)
print(pycode(series_T9))

phi_lim_series = series_T1 + series_T4 + series_T9

# 8. Both phi_lim_limit and phi_lim_series give the derivative d(phi2-phi1)/dx at dx=0.
# To reinforce the simplification (technique 3), we simplify the series result:
dphi_dx_simpl = sp.simplify(phi_lim_series)

# The x-component of the field is then given (with a minus sign convention):
Hx_sigma1 = -dphi_dx_simpl

# 9. simplification
# try simplify further
Hx_sigma1_improved = sp.factor(Hx_sigma1)
Hx_sigma1_improved = sp.trigsimp(Hx_sigma1_improved)
Hx_sigma1_improved = sp.simplify(Hx_sigma1_improved)

# 9. Display the final closed-form expression.
# The pycode conversion prints a Python-format string.
code_str = pycode(Hx_sigma1_improved)
print("Hx_sigma1_lim = " + code_str)


# Hx_sigma1_lim = (1/8)*sigma1*(a*math.log(-1/(b - math.sqrt(a**2 + b**2 + c**2))) + a*math.log(b + math.sqrt(a**2 + b**2 + c**2)) - 2*c*math.atan(a*b/(a**2 + c**2 + c*math.sqrt(a**2 + b**2 + c**2))))/(math.pi*mu0)
