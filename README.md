# SVS13A Astrochemical Analysis Pipeline

Python tools developed for the analysis of ALMA spectral cube data of the protobinary system **SVS13A**, focusing on the spatial and chemical segregation of complex organic molecules (COMs).

This repository contains the analysis pipeline used in the **2025 TARA Summer Research Project** supervised by **Dr. Tien-Hao Hsieh**, with guidance from **Dr. Yu-Nung Su (ASIAA)** and **Prof. Shih-Ping Lai (NTHU)**.

Author:  
Meng-Lun Wu  
Department of Chemistry  
National Taiwan University

---

# Scientific Motivation

Understanding how planetary systems form requires studying the chemical and dynamical evolution of **young stellar objects**.

The **Class I protostellar phase** is particularly important because:

- the envelope is dispersing
- the disk is still forming
- active accretion continues

In binary systems, **streamers** can transport material from the envelope to disks or connect two protostars.

These processes may produce **chemical segregation**, where different molecules trace different physical environments.

---

# Target Source

**SVS13A**

Location:
- NGC 1333 star-forming region
- Distance ≈ 300 pc

System properties:

- Class I protobinary
- Components: **VLA4A** and **VLA4B**
- Separation ≈ 90 AU

SVS13A is known to be rich in **complex organic molecules (COMs)** and is therefore an ideal laboratory for studying astrochemical processes.

---

# Observational Data

Data source:

**ALMA Band 6**

Frequency coverage:

218 – 233 GHz

Spectral windows:

10 spectral windows containing transitions of multiple COM species.

At the distance of SVS13A:
0.5 arcsec ≈ 150 AU

---

# Molecules Studied

O-bearing COMs:

- Glycolaldehyde
- Methyl formate
- Acetaldehyde
- Ethylene glycol

N-bearing species:

- Propanenitrile

These molecules are particularly useful because their **formation pathways differ**:

- grain-surface chemistry
- gas-phase reactions

Thus they can probe **thermal history and chemical evolution**.

---

# Pipeline Overview

The analysis pipeline performs several key tasks:

### 1 Spectral Cube Analysis

From ALMA data cubes, the pipeline generates:

- **Moment 0 maps** (integrated intensity)
- **Moment 1 maps** (velocity field)

These maps reveal:

- spatial distribution of molecular emission
- kinematic structures such as rotation or velocity gradients.

---

### 2 Position–Velocity (PV) Diagrams

PV diagrams are extracted along the axis connecting the two protostars.

These diagrams help visualize:

- rotational signatures
- velocity gradients
- possible infall or outflow motions

Keplerian reference curves are overlaid to compare observed structures with theoretical expectations.

---

### 3 Chemical Segregation Analysis

Moment maps reveal clear chemical differences:

Compact emission near **VLA4A**

- acetaldehyde
- ethylene glycol

More extended emission bridging toward **VLA4B**

- methyl formate
- propanenitrile

These spatial differences indicate **chemical segregation within the protobinary system**.

---

### 4 Joint Gaussian Spectral Fitting

To analyze spectral lines, the pipeline performs **joint Gaussian fitting** across multiple transitions.

Key features:

- multiple transitions share the same
  - centroid velocity
  - linewidth

This allows strong lines to constrain weaker ones.

Two models are tested:

- one-component Gaussian
- two-component Gaussian

Model selection uses the **Akaike Information Criterion (AIC)**.

---

### 5 Temperature Diagnostics

Gas temperatures are estimated using two methods:

#### Rotation Diagram Method

For molecules with ≥3 transitions:
ln(N_u / g_u) vs E_u

The slope gives the excitation temperature.

---

#### Line-Ratio Thermometry

For molecules with only two transitions:

temperature is derived from **line intensity ratios**.

---

# Key Results

1. **Chemical segregation is clearly observed**

O-bearing and N-bearing species show different spatial distributions.

2. **Velocity fields reveal rotational structures**

Moment 1 maps show a velocity gradient across **VLA4A** consistent with rotation.

3. **Temperature structure**

Acetaldehyde:
T ≈ 190 – 300 K

Glycolaldehyde:
T ≈ 35 – 75 K

This indicates molecule-dependent excitation conditions.

4. **Multiple gas components**

Two-component fits reveal:

- hot inner gas
- cooler envelope material

coexisting along the same line of sight.

---

# Example Outputs

The pipeline produces:

- moment maps
- PV diagrams
- spectral fits
- temperature diagnostics

Example outputs will be shown in the `figures/` directory.

---

# Dependencies

Python 3.10+

Main libraries:

- numpy
- scipy
- astropy
- matplotlib
- emcee
- spectral-cube

---

# Repository Structure
SVS13A-astrochemistry-analysis/

scripts/
moment_maps.py
pv_diagram.py
gaussian_fitting.py

src/
cube_utils.py
spectral_models.py
fitting_tools.py

notebooks/
analysis_demo.ipynb

figures/
example_maps.png

---

# Future Work

Next steps include:

- deriving absolute column densities
- optical-depth corrections
- abundance calculations
- comparison with astrochemical models

---

# Acknowledgements

This work was conducted during the **2025 TARA Summer Research Program**.

Supervisors:

- Dr. Tien-Hao Hsieh
- Dr. Yu-Nung Su
- Prof. Shih-Ping Lai
