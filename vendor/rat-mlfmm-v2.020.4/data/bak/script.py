import sympy as sp
from sympy.printing.pycode import pycode

# Define symbols and assume they are positive as needed.
a, b, c, dx, mu0, sigma1 = sp.symbols('a b c dx mu0 sigma1', real=True, positive=True)
pi = sp.pi

# For convenience, assume c > 0 so that abs(c) = c.
# Define the symbolic functions F0, F1, and F2.
# Note: These definitions are taken from your equations. 
#       Adjust the expressions if needed.
aa, bb, cc = sp.symbols('aa bb cc', real=True, positive=True)

F0_expr = 1/(4*pi*mu0)*(
    (aa/2)*sp.log((sp.sqrt(aa**2+bb**2+cc**2)+bb)/(sp.sqrt(aa**2+bb**2+cc**2)-bb))
    - cc * sp.atan((aa*bb)/( (aa**2+cc**2) + cc*sp.sqrt(aa**2+bb**2+cc**2) ))
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

# Create functions by substituting the generic variables with function arguments.
F0 = sp.Lambda((aa, bb, cc), sp.simplify(F0_expr))
F1 = sp.Lambda((aa, bb, cc), sp.simplify(F1_expr))
F2 = sp.Lambda((aa, bb, cc), sp.simplify(F2_expr))

# Define the shifted variables.
a1 = a - dx
b1 = b

# For the small triangles, define a2 and b2.
# (Here we assume a, b > 0 so that sqrt(a**2+b**2) is positive.)
a2 = (b/sp.sqrt(a**2 + b**2)) * dx
b2 = (a/sp.sqrt(a**2 + b**2)) * dx

# a3 and b3 for the second small triangle:
a3 = a2
b3 = sp.sqrt(a**2 + b**2)  # remains constant to leading order

# Geometric factors (from the dot products)
sigmaEP = -sigma1*(b/sp.sqrt(a**2+b**2))
sigmaOB = -sigma1*(a/sp.sqrt(a**2+b**2))

# Define the unshifted potential phi1 and the shifted potential phi2.
phi1 = sigma1 * F1(a, b, c)

phi2 = ( sigma1*dx*(F0(a1, b1, c) + F0(a2, b2, c) + F0(a3, b3, c))
         + sigma1*F1(a1, b1, c)
         + sigmaEP*( F1(a2, b2, c) + F1(a3, b3, c) )
         + sigmaOB*( F2(a2, b2, c) - F2(a3, b3, c) )
       )

# Compute the derivative with respect to dx.
dphi_dx = sp.diff(phi2 - phi1, dx)

# Instead of directly substituting dx=0, take the limit as dx -> 0.
dphi_dx_dx0 = sp.limit(dphi_dx, dx, 0)

# The x-component of the field is then:
Hx_sigma1 = -sp.simplify(dphi_dx_dx0)

# 8. Display the final closed-form expression for Hx_sigma1.
code_str2 = pycode(expr_factor_terms)
print("Hx_sigma1_lim = " + code_str2)



