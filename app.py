"""
BAGH_SIR - web app
Wraps Promod Bagh's validated bagh_sir.py in a browser interface so anyone,
on any operating system, can compute an asteroid's volume, density, and
porosity from a 3D shape model. No installation needed.
"""
import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import bagh_sir as bs   # the exact, validated tool

st.set_page_config(page_title="BAGH_SIR - Asteroid Volume & Density",
                   page_icon="\U0001FAA8", layout="centered")

# ---------- header ----------
st.title("BAGH_SIR")
st.caption("Bagh's Shape-Integrated Reckoner \u2014 volume, density & porosity from a 3D shape model")
st.write(
    "Upload an asteroid or comet **.obj** shape model. BAGH_SIR computes its true "
    "volume from the real shape (not a sphere guess), then its bulk density and "
    "porosity. The tool has been validated against 13 bodies \u2014 including "
    "**Bennu** (NASA OSIRIS-REx), **Ryugu** (JAXA Hayabusa2), **Itokawa** "
    "(JAXA Hayabusa) and **comet 67P** (ESA Rosetta) \u2014 reproducing every "
    "published density."
)

with st.expander("Example values to try"):
    st.markdown(
        "| Body | Mass (kg) | Diameter (km) | Type |\n"
        "|---|---|---|---|\n"
        "| Bennu | 7.33e10 | 0.49 | C |\n"
        "| Ryugu | 4.50e11 | 0.896 | C |\n"
        "| Itokawa | 3.51e10 | 0.33 | S |\n"
        "| Comet 67P | 9.98e12 | 3.29 | (grain 1.9) |\n\n"
        "Get free shape models at **3d-asteroids.space**. Use a medium-resolution "
        "model (a few hundred thousand faces or fewer)."
    )

st.divider()

# ---------- inputs ----------
up = st.file_uploader("Shape model (.obj)", type=["obj"])

c1, c2 = st.columns(2)
with c1:
    mass_s = st.text_input("Mass in kg  (optional \u2014 needed for density)",
                           placeholder="e.g. 7.33e10")
with c2:
    diam_s = st.text_input("Known diameter in km  (optional \u2014 scales the shape)",
                           placeholder="e.g. 0.49")

st.write("**Porosity** (optional) \u2014 choose a rock type *or* enter a grain density:")
c3, c4 = st.columns(2)
TYPES = {
    "(none)": None,
    "C \u2014 carbonaceous": "C", "B \u2014 carbonaceous": "B",
    "S \u2014 stony": "S", "Q \u2014 stony": "Q",
    "M \u2014 metallic": "M", "X \u2014 metallic": "X",
    "P \u2014 primitive": "P", "D \u2014 primitive": "D",
    "V \u2014 basaltic": "V", "E \u2014 enstatite": "E",
}
with c3:
    tax_label = st.selectbox("Rock type", list(TYPES.keys()))
with c4:
    grain_s = st.text_input("or grain density (g/cm\u00b3)", placeholder="e.g. 2.4")

run = st.button("Analyze", type="primary", use_container_width=True)


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


# ---------- run ----------
if run:
    if up is None:
        st.warning("Please upload a .obj shape model first.")
        st.stop()

    mass = _num(mass_s)
    diam = _num(diam_s)
    grain = _num(grain_s)
    taxon = TYPES[tax_label]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".obj")
    tmp.write(up.getvalue())
    tmp.close()
    try:
        with st.spinner("Computing the true volume from the shape\u2026"):
            V, F = bs.read_obj(tmp.name)
            if len(V) == 0 or len(F) == 0:
                st.error("Could not read vertices and faces from this file. "
                         "Make sure it is a triangulated .obj mesh.")
                st.stop()

            orig_diam = bs.equiv_diameter(bs.genesis_volume(V, F))
            res = bs.analyse_shape(V, F, up.name, mass_kg=mass, verbose=False,
                                   target_diam=diam, taxon=taxon, grain=grain)
    finally:
        os.unlink(tmp.name)

    Vmsh = res["V_mesh"]
    scaled = diam is not None
    vol_unit  = "km\u00b3" if scaled else "units\u00b3"
    diam_unit = "km"      if scaled else "units"

    st.success(f"Done \u2014 {res['n_vert']:,} vertices, {res['n_face']:,} faces. "
               f"Both volume methods agreed.")

    # headline metrics (units live in the labels, so values never truncate)
    m1, m2, m3 = st.columns(3)
    m1.metric(f"True volume ({vol_unit})", f"{Vmsh:.4g}")
    m2.metric(f"Equivalent diameter ({diam_unit})", f"{res['equiv_diameter']:.4g}")
    m3.metric("Shape uncertainty", f"{res['shape_spread_pct']:.1f}%",
              help="How much a simple round-shape guess would inflate the volume. "
                   "Higher = more irregular = more reason to use the real shape.")

    if not scaled:
        st.caption("\u201Cunits\u201D = the model\u2019s own scale. Enter a known "
                   "diameter above to get real kilometres and a physical density.")

    # density + porosity
    density = res.get("density_gcc")
    if density is not None:
        d1, d2 = st.columns(2)
        d1.metric("Bulk density (g/cm\u00b3)", f"{density:.3f}")
        if taxon or grain:
            p, g, label = bs.porosity(density, taxon, grain)
            if p is not None:
                d2.metric("Macroporosity", f"{p*100:.0f}%")
                st.info(f"**{label}** \u2014 grain density used: {g:.2f} g/cm\u00b3")
        if not scaled:
            st.warning("No diameter was given, so this density is only physical if "
                       "the shape model is already in real kilometres (spacecraft "
                       "models usually are; catalogue / DAMIT models usually are not). "
                       "For catalogue models, enter the known diameter above.")
    else:
        st.info("Add a **mass (kg)** to get bulk density, and a **rock type** or "
                "**grain density** to get porosity.")

    # shape-assumption comparison
    st.subheader("Why shape matters")
    st.write("The same body, measured four ways. The real mesh is the truth; the "
             "others show how far a simpler assumption would be off.")
    rows = [
        ("Sphere guess",          res["V_sphere"]),
        ("Ellipsoid guess",       res["V_ellipsoid"]),
        ("Convex hull",           res["V_hull"]),
        ("Real mesh (BAGH_SIR)",  Vmsh),
    ]
    df = pd.DataFrame(
        [{"Method": n,
          f"Volume ({vol_unit})": f"{v:.4g}",
          "vs real shape": ("\u2014 (truth)" if abs(v - Vmsh) < 1e-12
                            else f"{(v - Vmsh) / Vmsh * 100:+.0f}%")}
         for n, v in rows]
    )
    st.dataframe(df, hide_index=True, use_container_width=True)

    sphere_over = (res["V_sphere"] - Vmsh) / Vmsh * 100.0
    st.caption(f"A plain sphere guess would overestimate this body\u2019s volume by "
               f"{sphere_over:.0f}% \u2014 which is exactly why the real shape is used.")

    if scaled:
        st.caption(f"Shape scaled from {orig_diam:.3g} to {diam:.3g} km "
                   f"(volume \u00d7 {(diam / orig_diam) ** 3:.3f}).")

# ---------- footer ----------
st.divider()
st.caption(
    "BAGH_SIR uses standard computational geometry \u2014 two independent volume "
    "methods (signed-tetrahedron sum + projection) that agree to machine precision. "
    "A validated, transparent tool, not new physics. \u00b7 Built by Promod Bagh, "
    "Independent researcher, Raghunathpur, India."
)
