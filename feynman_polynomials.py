"""
feynman_polynomials.py
Curated battery of 26 polynomial equations used as a benchmark suite. Most
were originally drawn from, or intended to represent, the Feynman Symbolic
Regression Database (Udrescu & Tegmark, "AI Feynman", Science Advances 2020),
but a 2026-07-22 ID-level audit against the database's own official
FeynmanEquations.csv found this file's `label` strings did not reliably
identify the official equation they claimed to (see TODO.md item 9 for the
full per-entry verification). Every entry below was individually re-checked
against the primary source, with this outcome:

  - 4 entries keep their original ID, independently confirmed correct:
    I.34.27 (E = h*nu), I.14.3 (U = m*g*h), I.43.31 (Einstein relation), and
    I.13.4 (a same-form u=w=0 restriction of the official 3-velocity-
    component kinetic energy, not an exact match but a legitimate
    simplification of it).
  - 3 entries were relabeled to the *different* official ID their content
    actually matches: I.8.2 -> I.8.14 (squared form of the distance
    formula), I.12.3 -> I.12.2 (Coulomb's law), II.11.28 -> I.39.22 (ideal
    gas law).
  - The remaining 19 entries do not correspond to any official Feynman
    equation ID. Most are still real, recognizable physics relations (e.g.
    Newton's second law, angular momentum, the parallel-axis theorem), just
    not the equation their old ID claimed; two (`algebraic_cubic_toy`,
    `charge_force_toy`) have no real physical content and are kept only as
    polynomial-form stress cases. These are given descriptive, non-ID slugs
    instead of an invented "I.x.y"-style label, so nobody mistakes an
    unverified slug for a sourced database citation.

Only equations that are polynomial in all variables are included.
(Excludes I.6.2a, the Gaussian, which is not polynomial.)

Total: 26 equations.

Format: (label, expr_str, var_names, ranges_dict)

Sources:
- Udrescu & Tegmark, "AI Feynman", Science Advances 2020
- https://space.mit.edu/home/tegmark/aifeynman.html
- Official equation list cross-checked directly against the database's own
  FeynmanEquations.csv (mirrored at
  github.com/florianBachinger/FeynmanEquations-Python-JDIQ).

Notes:
- Equations involving trig, exp, log are excluded (not polynomial).
- Some equations involve a "dependent variable" that is already isolated
  (e.g. F = m*a becomes F - m*a = 0). We write the invariant form p(...) = 0.
- Ranges are chosen so sampling is efficient and numerically stable.
"""

feynman_polynomials = [

    # ── Degree 2 (quadratic) ──────────────────────────────────────────────

    # Corrected: repo content is really official I.8.14's distance formula,
    # squared (d = sqrt((x2-x1)^2+(y2-y1)^2)), not an "I.8.2" that exists in
    # the official 100.
    ("I.8.14",   "(x2-x1)**2 + (y2-y1)**2 - d**2", ["x1","y1","x2","y2","d"],
                 {"x1":(-2,2),"y1":(-2,2),"x2":(-2,2),"y2":(-2,2),"d":(0,6)}),

    # Not an official Feynman ID (verified); a plain circle locus.
    ("circle_locus", "x**2 + y**2 - r**2",  ["x","y","r"],
                 {"x":(-2,2),"y":(-2,2),"r":(0,3)}),

    # Not an official Feynman ID (verified); Newton's second law.
    ("newtons_second_law", "F - m*a",             ["F","m","a"],
                 {"F":(-20,20),"m":(0.5,5),"a":(-5,5)}),

    # Not an official Feynman ID (verified); 1D momentum conservation.
    ("momentum_conservation_1d", "m1*v1 + m2*v2 - P",   ["m1","v1","m2","v2","P"],
                 {"m1":(0.5,2),"v1":(-2,2),"m2":(0.5,2),"v2":(-2,2),"P":(-5,5)}),

    # Not an official Feynman ID (verified); work-energy relation W = F*d.
    ("work_energy", "W - F*d",             ["W","F","d"],
                 {"W":(-20,20),"F":(-10,10),"d":(-3,3)}),

    # Not an official Feynman ID (verified); non-relativistic momentum
    # definition p = m*v.
    ("momentum_definition", "p - m*v",             ["p","m","v"],
                 {"p":(-10,10),"m":(0.5,5),"v":(-5,5)}),

    # Not an official Feynman ID (verified); 2D angular momentum.
    ("angular_momentum_2d", "x*vy - y*vx - L",     ["x","y","vx","vy","L"],
                 {"x":(-2,2),"y":(-2,2),"vx":(-2,2),"vy":(-2,2),"L":(-5,5)}),

    # Not an official Feynman ID (verified); power definition P = F*v.
    ("power_definition", "P - F*v",             ["P","F","v"],
                 {"P":(-30,30),"F":(-10,10),"v":(-5,5)}),

    # Not an official Feynman ID (verified); linear frequency-shift relation.
    ("frequency_shift_linear", "f - f0 - v*f",        ["f","f0","v"],
                 {"f":(1,10),"f0":(1,10),"v":(-0.9,0.9)}),

    # Not an official Feynman ID (verified); linear wavenumber-dispersion
    # relation.
    ("wavenumber_dispersion", "omega - omega0 - v*k", ["omega","omega0","v","k"],
                 {"omega":(-10,10),"omega0":(-5,5),"v":(-3,3),"k":(-3,3)}),

    # Not an official Feynman ID (verified); 3D Euclidean distance.
    ("distance_3d", "x**2 + y**2 + z**2 - d**2", ["x","y","z","d"],
                 {"x":(-3,3),"y":(-3,3),"z":(-3,3),"d":(0,5)}),

    # Corrected: repo content is really official I.39.22's ideal gas law
    # (Pr = n*kb*T/V), not an "II.11.28" that exists in the official 100
    # (whose actual content is relative permittivity).
    ("I.39.22",  "p*V - n*T",           ["p","V","n","T"],
                 {"p":(1,5),"V":(1,5),"n":(1,3),"T":(1,5)}),

    # ── Planck relation and its (non-official) duplicate ────────────────
    # I.34.27 is independently confirmed correct: E = h*nu, algebraically
    # the same relation as the official h/(2*pi)*omega form.
    ("I.34.27",  "E - h*nu",            ["E","h","nu"],
                 {"E":(1,20),"h":(0.5,5),"nu":(1,10)}),

    # Not an official Feynman ID (verified): the official I.41.16 is the
    # transcendental Planck blackbody radiance law, not E = h*nu. This entry
    # is algebraically the same relation as I.34.27 above, kept as a
    # separate duplicate entry for historical continuity with earlier runs.
    ("planck_relation_duplicate", "h*nu - E",            ["E","h","nu"],
                 {"E":(1,20),"h":(0.5,5),"nu":(1,10)}),

    # ── Degree 3 (cubic) ──────────────────────────────────────────────────

    # Corrected: repo content is really official I.12.2's Coulomb's law
    # (F = q1*q2/(4*pi*eps*r^2), multiplied through by r^2), not an
    # "I.12.3" that exists in the official 100.
    ("I.12.2",   "F*r**2 - k*q1*q2",    ["F","r","k","q1","q2"],
                 {"F":(-20,20),"r":(0.5,3),"k":(1,2),"q1":(-2,2),"q2":(-2,2)}),

    # Not an official Feynman ID (verified); centripetal force.
    ("centripetal_force", "r*F - m*v**2",        ["r","F","m","v"],
                 {"r":(0.5,3),"F":(0,30),"m":(0.5,5),"v":(-3,3)}),

    # Not an official Feynman ID (verified). Also not a dimensionally real
    # Coulomb-type law (force proportional to r, not 1/r^2); kept only as a
    # polynomial-form stress case, not as physics.
    ("charge_force_toy", "F - q1*q2*r",         ["F","q1","q2","r"],
                 {"F":(-20,20),"q1":(-2,2),"q2":(-2,2),"r":(0.5,3)}),

    # I.13.4 is independently confirmed correct: the official 1D kinetic
    # energy formula restricted to a single velocity component (the official
    # K = (1/2)*m*(v^2+u^2+w^2) with u=w=0).
    ("I.13.4",   "KE - m*v**2/2",       ["KE","m","v"],
                 {"KE":(0,20),"m":(0.5,5),"v":(-3,3)}),

    # Not an official Feynman ID (verified); a rescaled duplicate of I.13.4
    # above (2*KE = m*v^2).
    ("kinetic_energy_scaled", "2*KE - m*v**2",       ["KE","m","v"],
                 {"KE":(0,20),"m":(0.5,5),"v":(-3,3)}),

    # I.14.3 is independently confirmed correct: U = m*g*h (h/z renamed).
    ("I.14.3",   "U - m*g*h",           ["U","m","g","h"],
                 {"U":(-50,50),"m":(1,5),"g":(5,15),"h":(0,5)}),

    # Not an official Feynman ID (verified); 1D kinematics position formula.
    ("kinematics_position", "x - u*t - a*t**2/2",   ["x","u","t","a"],
                 {"x":(-10,10),"u":(-3,3),"t":(0,3),"a":(-2,2)}),

    # Not an official Feynman ID (verified); the parallel-axis theorem.
    ("parallel_axis_theorem", "I - Ic - m*d**2",     ["I","Ic","m","d"],
                 {"I":(0,30),"Ic":(0,10),"m":(0.5,5),"d":(0,3)}),

    # Not an official Feynman ID (verified); gravitational potential energy
    # of a two-body orbit.
    ("gravitational_pe_orbit", "U*r + G*m1*m2",       ["U","r","G","m1","m2"],
                 {"U":(-50,-0.1),"r":(0.5,5),"G":(1,2),"m1":(0.5,3),"m2":(0.5,3)}),

    # Not an official Feynman ID (verified). No evident physical content;
    # kept only as a polynomial-form stress case.
    ("algebraic_cubic_toy", "x**3 + 2*x*y + y**2", ["x","y"],
                 {"x":(-2,2),"y":(-2,2)}),

    # Not an official Feynman ID (verified); a polarization-correction-style
    # relation.
    ("polarization_correction", "n*kT - n0*kT - n0*p*Ef", ["n","n0","p","Ef","kT"],
                 {"n":(1,20),"n0":(1,5),"p":(-1,1),"Ef":(-2,2),"kT":(0.1,2)}),

    # I.43.31 is independently confirmed correct: the Einstein relation
    # D = mob*kb*T.
    ("I.43.31",  "D - k*T*mu",          ["D","k","T","mu"],
                 {"D":(0,20),"k":(1,2),"T":(1,5),"mu":(0.5,5)}),
]
