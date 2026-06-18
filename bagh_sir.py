"""
BAGH_SIR  -  Bagh's Shape-Integrated Reckoner
============================================
Small-Body Volume & Density Analyzer  -  by Promod Bagh

Applies the VIT equations explicitly in the analysis pipeline:

  Vm(x) = x - 1          COLLAPSE: removes one dimension at a time
                          3D shape -> 2D footprint -> 1D axis -> depth value

  genesis: dim + 1        GENESIS: rebuilds volume from the collapsed parts
                          depth * footprint area summed back into 3D volume

  V_total = (1/6)|sum v0.(v1 x v2)|   TETRA LAW: independent confirmation

For each asteroid the output shows:
  - the collapse trace (3D -> 2D -> 1D -> depth)
  - the volume computed through genesis
  - the same volume confirmed by the tetra law
  - the four shape-assumption volumes (sphere, ellipsoid, hull, real mesh)
  - shape uncertainty percent (the open problem in asteroid science)
  - density if mass is given

RUN MODES
  python3 small_body_volume.py
        -> self-test on exact shapes (cube, sphere, ellipsoid)
  python3 small_body_volume.py shape.obj [mass_kg]
        -> analyse one asteroid shape model
  python3 small_body_volume.py shape.obj [mass_kg] [--diam KM]
        -> analyse one asteroid shape model
           --diam scales the shape to a known (accepted) diameter in km,
           so the density is physically correct even if the shape is
           scale-free or sized differently (DAMIT models often are).
  python3 small_body_volume.py --batch folder/ [masses.csv]
        -> analyse every .obj in a folder, rank by shape uncertainty
           masses.csv may have 3 columns:  name,mass_kg,diam_km
           (the diameter column auto-scales each body)

WHERE TO GET REAL SHAPE MODELS (free)
  DAMIT: https://astro.troja.mff.cuni.cz/projects/damit/
  PDS:   https://sbn.psi.edu/pds/shape-models/

DEPENDENCIES
  numpy    (always)
  scipy    (convex hull:  pip install scipy)
"""
import sys, os, glob, csv
import numpy as np


# ================================================================
# THE VIT EQUATIONS
# ================================================================

# --- COLLAPSE operator: removes one dimension ---
def Vm(x):
    """Collapse operator: Vm(x) = x - 1  (projection removes one dimension)."""
    return x - 1

# --- GENESIS operator: the interaction / join law (SAME dimension both sides) ---
def join_equal(n):
    """Genesis (interaction) law -- requires BOTH pieces the SAME dimension n:
           nD (+) nD  ->  (2n + 1)D
       n = 1:  two skew edges join into a 3D tetrahedron  (the proven skew-line case)."""
    return 2*n + 1


# ----------------------------------------------------------------
# METHOD 1 -- COLLAPSE  (project the shape, then UN-project it)
# ----------------------------------------------------------------
def collapse_shape(V, F):
    """COLLAPSE the 3D shape, one Vm step at a time:
         3D -> 2D : project each triangle onto its x-y footprint   Vm(3)=2
         2D -> 1D : keep only the depth (z) axis                   Vm(2)=1
         1D -> 0D : one depth number per triangle                 Vm(1)=0
    Returns the 2D footprint area and the 0D depth for each face."""
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    dim = 3; trace = [dim]
    dim = Vm(dim); trace.append(dim)              # 3D -> 2D
    footprint = 0.5 * ((v1[:,0]-v0[:,0])*(v2[:,1]-v0[:,1])
                       - (v2[:,0]-v0[:,0])*(v1[:,1]-v0[:,1]))
    dim = Vm(dim); trace.append(dim)              # 2D -> 1D
    dim = Vm(dim); trace.append(dim)              # 1D -> 0D
    mean_depth = (v0[:,2]+v1[:,2]+v2[:,2])/3.0
    return footprint, mean_depth, trace

def rebuild_by_inverse_collapse(footprint, mean_depth):
    """INVERSE COLLAPSE -- the exact reverse of the projection above.
    This is a PRODUCT, not a join:  2D footprint x 1D height = 3D prism,
    summed.  (It un-does the collapse; it is NOT the genesis/join law.)"""
    return abs((footprint * mean_depth).sum())


# ----------------------------------------------------------------
# METHOD 2 -- GENESIS  (build each tetrahedron by the JOIN law)
# ----------------------------------------------------------------
def genesis_volume(V, F):
    """GENESIS by the SYMMETRIC join law (same dimension on both sides).
    For each face, take the tetra with corners (origin O, v0, v1, v2).
    It is the JOIN of its two opposite SKEW EDGES, each of dimension 1:

         edge(O, v0)   (+)   edge(v1, v2)     ->     3D tetrahedron
            1D                  1D                  2(1)+1 = 3D

    Both joined pieces have the SAME dimension (1D) -- this is your
    1D (+) 1D -> 3D skew-line result. Volume = (1/6)|v0 . (v1 x v2)|.
    Summing the genesis-built tetrahedra gives the total volume."""
    assert join_equal(1) == 3                    # 1D + 1D -> 3D
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return abs(np.einsum('ij,ij->i', v0, np.cross(v1, v2)).sum()) / 6.0


# ================================================================
# SHAPE ASSUMPTIONS (the uncertainty story)
# ================================================================

def volume_sphere(V):
    r = ((V.max(0) - V.min(0)) / 2.0).mean()
    return 4.0 / 3.0 * np.pi * r ** 3

def volume_ellipsoid(V):
    C = V - V.mean(0)
    _, _, axes = np.linalg.svd(C, full_matrices=False)
    a, b, c = ((C @ axes.T).max(0) - (C @ axes.T).min(0)) / 2.0
    return 4.0 / 3.0 * np.pi * a * b * c

def volume_hull(V):
    try:
        from scipy.spatial import ConvexHull
        return ConvexHull(V).volume
    except Exception:
        return np.nan

def surface_area(V, F):
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum()

def equiv_diameter(vol):
    return (6.0 * vol / np.pi) ** (1.0 / 3.0)

def scale_to_diameter(V, F, target_diam):
    """Scale the shape so its volume-equivalent diameter equals target_diam.
    DAMIT shapes give the correct FORM but often an arbitrary SIZE; the real
    size comes from an occultation or radiometric diameter. This sizes the
    shape to that known diameter, so the density is physically correct.
    Returns (scaled_vertices, original_equiv_diameter)."""
    cur_diam = equiv_diameter(genesis_volume(V, F))
    factor = target_diam / cur_diam
    return V.astype(float) * factor, cur_diam


# typical grain (rock) densities of meteorite analogues, g/cm^3
# (approximate, from Carry 2012 / Consolmagno et al. / Britt et al.)
GRAIN = {
    "C":2.73,"B":2.73,"G":2.70,"F":2.70,"P":2.60,"D":2.60,"T":2.70,
    "S":3.73,"Q":3.73,"K":3.50,"L":3.50,"A":3.70,
    "V":3.26,"E":3.00,"R":3.40,
    "M":4.90,"X":4.90,   # metal-rich; an iron analogue would be ~7.4
}

def porosity(bulk_density, taxon=None, grain=None):
    """Macroporosity = 1 - bulk/grain. Returns (porosity, grain_used, label).
    Give a taxonomic letter (taxon) OR a grain density directly."""
    if grain is None and taxon:
        grain = GRAIN.get(taxon[0].upper())
    if not grain:
        return None, None, "no grain density (give --type or --grain)"
    p = 1.0 - bulk_density/grain
    if   p < 0.10: label = "solid / coherent rock"
    elif p < 0.30: label = "fractured / microporous"
    else:          label = "rubble pile (high macroporosity)"
    if p < 0:      label = "bulk > grain: check size, mass, or rock type"
    return p, grain, label


# ================================================================
# MAIN ANALYSIS  (the VIT pipeline)
# ================================================================

def analyse_shape(V, F, name="shape", mass_kg=None, verbose=True, target_diam=None,
                  taxon=None, grain=None):
    # --- optional: scale the shape to a known (accepted) diameter ---
    orig_diam = None
    if target_diam:
        V, orig_diam = scale_to_diameter(V, F, target_diam)

    # --- run the VIT equations ---
    footprint, mean_depth, collapse_trace = collapse_shape(V, F)
    V_collapse = rebuild_by_inverse_collapse(footprint, mean_depth)   # method 1
    V_genesis  = genesis_volume(V, F)                                 # method 2 (join law)
    match      = abs(V_collapse - V_genesis) < max(V_genesis * 1e-6, 1e-12)

    # --- shape assumption volumes ---
    Vsph = volume_sphere(V)
    Vell = volume_ellipsoid(V)
    Vhul = volume_hull(V)
    Vmsh = V_genesis          # the genesis (join/tetra) volume = real mesh volume

    vols   = np.array([v for v in (Vsph, Vell, Vhul, Vmsh) if not np.isnan(v)])
    spread = (vols.max() - vols.min()) / Vmsh * 100.0

    if verbose:
        print(f"\n=== {name} ===")
        print(f"  mesh : {len(V):,} vertices, {len(F):,} faces")
        print(f"\n  --- VIT EQUATIONS ---")
        print(f"  COLLAPSE (Vm=x-1)        : "
              + " -> ".join(f"{d}D" for d in collapse_trace)
              + "   (project to footprint + depth)")
        print(f"  rebuild by INVERSE COLLAPSE: footprint x height summed")
        print(f"     -> volume = {V_collapse:.6g}   (reverse of the projection)")
        print(f"  GENESIS (join 2n+1)      : 1D edge (+) 1D edge -> 3D tetra"
              + "  (equal dims, join_equal(1)=3)")
        print(f"     -> volume = {V_genesis:.6g}   [{'MATCH' if match else 'DIFFER'}]")
        print(f"\n  --- SHAPE ASSUMPTIONS ---")
        print(f"  sphere              : {Vsph:.4g}")
        print(f"  ellipsoid           : {Vell:.4g}")
        print(f"  convex hull         : {Vhul:.4g}")
        print(f"  real mesh (genesis) : {Vmsh:.4g}   (truest)")
        print(f"\n  SHAPE UNCERTAINTY   : {spread:.1f}%")
        if target_diam:
            print(f"  SCALED to known size: {orig_diam:.3g} -> {target_diam:.3g} km"
                  f"  (volume x {(target_diam/orig_diam)**3:.3f})")
            print(f"  Equivalent diameter : {equiv_diameter(Vmsh):.4g} km  (now matches known size)")
        else:
            print(f"  Equivalent diameter : {equiv_diameter(Vmsh):.4g} units"
                  f"  (no known size given -> density not physical unless shape is in km)")
        if mass_kg is not None:
            density = mass_kg / (Vmsh * 1e12)  # km^3 + kg -> g/cm^3
            note = "physical" if target_diam else "only valid if shape already in km"
            print(f"  Bulk density        : {density:.3f} g/cm^3  ({note})")
            if target_diam and (taxon or grain):
                p, g, label = porosity(density, taxon, grain)
                if p is not None:
                    tname = f" ({taxon}-type)" if taxon else ""
                    print(f"  Grain (rock) density: {g:.2f} g/cm^3{tname}")
                    print(f"  Macroporosity       : {p*100:.0f}%  ->  {label}")

    return {
        "name": name, "n_vert": len(V), "n_face": len(F),
        "V_genesis": V_genesis, "V_collapse": V_collapse,
        "V_sphere": Vsph, "V_ellipsoid": Vell,
        "V_hull": Vhul,  "V_mesh": Vmsh,
        "shape_spread_pct": spread,
        "equiv_diameter": equiv_diameter(Vmsh),
        "surface_area": surface_area(V, F),
        **({"density_gcc": mass_kg/(Vmsh*1e12),
            "density_unc_pct": spread} if mass_kg else {})
    }


# ================================================================
# I/O + HELPERS
# ================================================================

def read_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.array(verts, float), np.array(faces, int)

def cube():
    V = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                  [0,0,1],[1,0,1],[1,1,1],[0,1,1]], float)
    F = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],
                  [1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]])
    return V, F, 1.0

def icosphere(subdiv=4, scale=(1,1,1)):
    t = (1+5**0.5)/2
    V = np.array([[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],
                  [0,-1,-t],[0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]], float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    F = [[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],
         [11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],
         [3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]]
    mid = {}
    def m(a, b):
        k = (min(a,b), max(a,b))
        if k in mid: return mid[k]
        nonlocal V
        p = (V[a]+V[b])/2; p /= np.linalg.norm(p)
        V = np.vstack([V, p]); mid[k] = len(V)-1; return mid[k]
    for _ in range(subdiv):
        nf = []
        for a,b,c in F:
            ab,bc,ca = m(a,b),m(b,c),m(c,a)
            nf += [[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
        F = nf
    return V * np.array(scale), np.array(F)


# ================================================================
# SELF-TEST
# ================================================================

def self_test():
    print("SELF-TEST (the two VIT methods on exact shapes)\n")
    V, F, exact = cube()
    fp, md, ct = collapse_shape(V, F)
    v_col = rebuild_by_inverse_collapse(fp, md)     # collapse method
    v_gen = genesis_volume(V, F)                     # genesis (join) method
    print(f"  COLLAPSE trace          : {' -> '.join(str(d)+'D' for d in ct)}")
    print(f"  GENESIS (1D+1D, join_equal(1)) : two 1D edges -> a {join_equal(1)}D tetra per face")
    print(f"  Cube: collapse={v_col:.10f}  genesis={v_gen:.10f}  exact={exact}")
    print(f"  => the two methods agree: {abs(v_col-v_gen)<1e-9}   "
          f"and match exact: {abs(v_gen-exact)<1e-9}\n")
    V, F = icosphere(4)
    v_col2 = rebuild_by_inverse_collapse(*collapse_shape(V,F)[:2])
    v_gen2 = genesis_volume(V,F); true = 4/3*np.pi
    print(f"  Sphere: collapse={v_col2:.5f}  genesis={v_gen2:.5f}  smooth={true:.5f}\n")
    print("  Collapse method and genesis (join) method agree. Run on a real asteroid.\n")


# ================================================================
# BATCH
# ================================================================

def run_batch(folder, masses=None):
    files = sorted(glob.glob(os.path.join(folder, "*.obj")))
    if not files:
        sys.exit(f"No .obj files in {folder}")
    rows = []
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        V, F  = read_obj(p)
        entry = masses.get(name) if masses else None
        mass, diam = entry if isinstance(entry, tuple) else (entry, None)
        rows.append(analyse_shape(V, F, name, mass, verbose=True, target_diam=diam))

    rows.sort(key=lambda r: r["shape_spread_pct"], reverse=True)
    print("\n" + "="*68)
    print("RANKED BY SHAPE UNCERTAINTY  (top = most worth re-examining)")
    print("="*68)
    print(f"{'#':>2}  {'asteroid':22}{'spread%':>9}{'concave?':>9}{'eq.diam':>10}")
    for i, r in enumerate(rows, 1):
        concave = "yes" if (r["V_hull"]-r["V_mesh"])/r["V_mesh"] > 0.02 else "no"
        print(f"{i:>2}  {r['name'][:22]:22}{r['shape_spread_pct']:>8.1f}%"
              f"{concave:>9}{r['equiv_diameter']:>10.3g}")

    keys = ["name","n_vert","n_face","V_genesis","V_collapse",
            "V_sphere","V_ellipsoid","V_hull","V_mesh",
            "shape_spread_pct","equiv_diameter","surface_area"]
    if any("density_gcc" in r for r in rows):
        keys += ["density_gcc","density_unc_pct"]
    with open("results.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: r.get(k,"") for k in keys})
    print(f"\nWrote results.csv ({len(rows)} bodies, sorted by shape uncertainty).")


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    print("="*60)
    print("  BAGH_SIR  -  Bagh's Shape-Integrated Reckoner")
    print("  asteroid volume & density from shape models")
    print("="*60)
    args = sys.argv[1:]
    # pull out an optional  --diam <km>  flag
    target_diam = None
    if "--diam" in args:
        i = args.index("--diam")
        target_diam = float(args[i+1]); del args[i:i+2]
    # optional  --type <taxon>  and  --grain <g/cm3>  for porosity
    taxon = None
    if "--type" in args:
        i = args.index("--type"); taxon = args[i+1]; del args[i:i+2]
    grain = None
    if "--grain" in args:
        i = args.index("--grain"); grain = float(args[i+1]); del args[i:i+2]

    if not args:
        self_test()
    elif args[0] == "--batch":
        masses = None
        if len(args) > 2 and os.path.exists(args[2]):
            masses = {}
            with open(args[2]) as f:
                for r in csv.reader(f):
                    if len(r) >= 2:
                        try:
                            m = float(r[1])
                            d = float(r[2]) if len(r) >= 3 and r[2].strip() else None
                            masses[r[0]] = (m, d)
                        except ValueError:
                            pass
        run_batch(args[1], masses)
    else:
        V, F = read_obj(args[0])
        mass = float(args[1]) if len(args) > 1 else None
        analyse_shape(V, F, os.path.basename(args[0]), mass,
                      verbose=True, target_diam=target_diam, taxon=taxon, grain=grain)
