//! EMCraft's templates, all in one place: the models the start page and `file.sample` open,
//! each a command script (`scripts/*.json`) whose dimensions are project variables (edited
//! with `project.variable`, which rebuilds the model), and the geometry builders those scripts
//! call when a dimension is a count (the iris filter's irises).
//!
//! A template is added here and nowhere else: its script in `scripts/`, where its parameters
//! show on the model in `scripts/<name>.marks.json`, a row in [`TEMPLATES`], and a builder
//! module when its shape needs one. A template whose shape is a SolidCraft design carries the
//! design in `designs/` (a row in [`DESIGNS`]); its script opens it by `geometry.import` with
//! `path` `template:<file>` ([`design`]), the template's parameters passed to the design's.
//!
//! ```
//! let t = emcraft_templates::find("iris-bandpass").map(|t| t.name);
//! assert_eq!(t, Some("iris-bandpass"));
//! assert!(emcraft_templates::find("nothing").is_none());
//! ```

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

mod iris;

pub use emcraft_model::Analysis;
pub use iris::{IrisFilter, MAX_IRISES, iris_sections};

/// A template.
pub struct Template {
    pub name: &'static str,
    /// What it is in a few words (a card's title).
    pub title: &'static str,
    /// What it is in full: its sizes, materials and what it shows.
    pub label: &'static str,
    /// The command script (a JSON array of `{"command", "params"}`).
    pub script: &'static str,
    /// Where each parameter shows on the model: `{parameter: {"from": [x, y, z], "to": [x, y,
    /// z]}}`, a dimension between two points in the display units, each coordinate a number or
    /// an expression of the parameters (`"=(a+w)/2"`); `{}` for none.
    pub marks: &'static str,
    /// The kinds of analysis it is made for, the first its own (the start page lists it under
    /// each).
    pub kinds: &'static [Analysis],
    /// What it models: the start page's heading it shows under.
    pub group: Group,
}

/// What a template models, in the start page's order: the headings its lists fall under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Group {
    /// Filters: high-pass, band-pass, low-pass and notch.
    Filters,
    /// Guides, lines, tapers and matching sections.
    Lines,
    /// Bends, tees, steps and posts: a guide's discontinuities.
    Junctions,
    /// Directional couplers and power dividers.
    Couplers,
    /// Loads, material samples, slabs and ferrites.
    Materials,
    /// Periodic guides and pulse compressors.
    Periodic,
    /// Cavities and resonators.
    Resonators,
    /// Bloch bands of a period.
    Dispersion,
    /// Antennas.
    Antennas,
    /// Plane waves on objects.
    Scattering,
    /// Board checks: routes, returns and coupling.
    Boards,
    /// Chains of stages from an antenna to a receiver: their budget.
    Systems,
    /// Electrons in their own fields: diodes, plasmas and beams (particle-in-cell).
    Particles,
}

impl Group {
    /// Every group, in the start page's order.
    pub const ALL: [Group; 13] = [
        Group::Filters,
        Group::Lines,
        Group::Junctions,
        Group::Couplers,
        Group::Materials,
        Group::Periodic,
        Group::Resonators,
        Group::Dispersion,
        Group::Antennas,
        Group::Scattering,
        Group::Boards,
        Group::Systems,
        Group::Particles,
    ];

    /// Its name in `file.templates`' replies.
    ///
    /// ```
    /// assert_eq!(emcraft_templates::Group::Junctions.id(), "junctions");
    /// ```
    pub fn id(self) -> &'static str {
        match self {
            Group::Filters => "filters",
            Group::Lines => "lines",
            Group::Junctions => "junctions",
            Group::Couplers => "couplers",
            Group::Materials => "materials",
            Group::Periodic => "periodic",
            Group::Resonators => "resonators",
            Group::Dispersion => "dispersion",
            Group::Antennas => "antennas",
            Group::Scattering => "scattering",
            Group::Boards => "boards",
            Group::Systems => "systems",
            Group::Particles => "particles",
        }
    }
}

pub const TEMPLATES: &[Template] = &[
    // Filters.
    Template {
        name: "highpass",
        title: "Waveguide high-pass",
        label: "Waveguide high-pass filter: WR-90 filled with εr = 2 on both sides of a 20 mm empty section, cut off below 6.56 GHz",
        script: include_str!("../scripts/highpass.json"),
        marks: include_str!("../scripts/highpass.marks.json"),
        kinds: &[Analysis::Frequency, Analysis::TimeDomain],
        group: Group::Filters,
    },
    Template {
        name: "iris-bandpass",
        title: "Iris band-pass filter",
        label: "Inductive-iris band-pass filter: three WR-90 cavities between four 2 mm irises (12.75, 9.25, 9.25, 12.75 mm), passband 9.75–10.25 GHz",
        script: include_str!("../scripts/iris-bandpass.json"),
        marks: include_str!("../scripts/iris-bandpass.marks.json"),
        kinds: &[Analysis::Frequency, Analysis::TimeDomain],
        group: Group::Filters,
    },
    Template {
        name: "coupled-filter",
        title: "Coupled-line band-pass",
        label: "Parallel-coupled stripline band-pass filter: three half-wave resonators coupled edge to edge between 50 Ω feeds, PTFE between planes 4 mm apart, 4.66–5.34 GHz (Cohn 1958)",
        script: include_str!("../scripts/coupled-filter.json"),
        marks: "{}",
        kinds: &[Analysis::Frequency],
        group: Group::Filters,
    },
    Template {
        name: "pcb-stub-filter",
        title: "Stub notch filter",
        label: "Microstrip open-stub notch filter on 0.508 mm of εr = 3.55: a quarter-wave stub off a 50 Ω line, its notch near 5 GHz",
        script: include_str!("../scripts/pcb-stub-filter.json"),
        marks: "{}",
        kinds: &[Analysis::Frequency],
        group: Group::Filters,
    },
    Template {
        name: "pcb-lowpass",
        title: "Stepped-impedance low-pass",
        label: "Stepped-impedance microstrip low-pass filter: three narrow high-impedance lines between wide low-impedance pads, cut off near 5.5 GHz",
        script: include_str!("../scripts/pcb-lowpass.json"),
        marks: "{}",
        kinds: &[Analysis::Frequency],
        group: Group::Filters,
    },
    Template {
        name: "adsb-bandpass",
        title: "ADS-B band-pass",
        label: "ADS-B band-pass filter at 1090 MHz on 1.6 mm of FR-4: three gap-coupled half-wave microstrip resonators (third-order Chebyshev, 0.5 dB ripple) with chip capacitors across the gaps: about 2.7 dB through its band from 1.02 to 1.16 GHz (the FR-4's loss), 17 dB down at 0.98 GHz",
        script: include_str!("../scripts/adsb-bandpass.json"),
        marks: "{}",
        kinds: &[Analysis::Frequency, Analysis::Pcb],
        group: Group::Filters,
    },
    // Lines.
    Template {
        name: "straight",
        title: "Straight waveguide",
        label: "Straight rectangular waveguide, WR-90 (a × b = 22.86 × 10.16 mm) by default, 50 mm long, swept over its band (1.25 to 1.89 times the TE10 cut-off): set a and b to another EIA size (WR-62: 15.80 × 7.90)",
        script: include_str!("../scripts/straight.json"),
        marks: include_str!("../scripts/straight.marks.json"),
        kinds: &[Analysis::Frequency],
        group: Group::Lines,
    },
    Template {
        name: "circular-guide",
        title: "Circular waveguide",
        label: "Circular waveguide, radius 10 mm, 30 mm long, on an unstructured mesh: TE11 (cut-off 8.79 GHz) through ports closed by the Bessel modes",
        script: include_str!("../scripts/circular-guide.json"),
        marks: include_str!("../scripts/circular-guide.marks.json"),
        kinds: &[Analysis::Frequency],
        group: Group::Lines,
    },
    Template {
        name: "coaxial-line",
        title: "Coaxial line",
        label: "PTFE coaxial line, conductors of radii 1 and 3.5 mm, 20 mm long: TEM through ports whose modes are solved on the annulus",
        script: include_str!("../scripts/coaxial-line.json"),
        marks: include_str!("../scripts/coaxial-line.marks.json"),
        kinds: &[Analysis::Frequency],
        group: Group::Lines,
    },
    Template {
        name: "taper",
        title: "Waveguide taper",
        label: "Linear taper from WR-90 to WR-75 (22.86 × 10.16 to 19.05 × 9.525 mm) over 40 mm, a SolidCraft design: its small reflection over the band both guides share, against the theory of small reflections",
        script: include_str!("../scripts/taper.json"),
        marks: include_str!("../scripts/taper.marks.json"),
        kinds: &[Analysis::Frequency],
        group: Group::Lines,
    },
    Template {
        name: "td-quarter-wave",
        title: "Quarter-wave match",
        label: "WR-90 into a dielectric load (εr = 2.2) matched by a quarter-wave section of permittivity √εr, in the time domain",
        script: include_str!("../scripts/td-quarter-wave.json"),
        marks: "{}",
        kinds: &[Analysis::TimeDomain],
        group: Group::Lines,
    },
    Template {
        name: "td-wall-loss",
        title: "Lossy walls",
        label: "200 mm of WR-90 with walls of conductivity 1 MS/m in the time domain: its attenuation through the walls' surface impedance",
        script: include_str!("../scripts/td-wall-loss.json"),
        marks: "{}",
        kinds: &[Analysis::TimeDomain],
        group: Group::Lines,
    },
    Template {
        name: "pcb-quarter-wave",
        title: "Quarter-wave transformer",
        label: "Microstrip quarter-wave transformer from 50 Ω to 100 Ω through a 70 Ω section l_q long; port 2 at 100 Ω",
        script: include_str!("../scripts/pcb-quarter-wave.json"),
        marks: "{}",
        kinds: &[Analysis::Frequency],
        group: Group::Lines,
    },
    // Junctions.
    Template {
        name: "h-bend",
        title: "H-plane bend",
        label: "H-plane 90° bend in WR-90, its outer corner mitred by m on each wall (0: a square corner); ports on the arms' ends across y and x",
        script: include_str!("../scripts/h-bend.json"),
        marks: include_str!("../scripts/h-bend.marks.json"),
        kinds: &[Analysis::Frequency],
        group: Group::Junctions,
    },
    Template {
        name: "e-bend",
        title: "E-plane bend",
        label: "E-plane 90° bend in WR-90, its outer corner mitred by m on each wall (0: a square corner); ports on the arms' ends across y and x",
        script: include_str!("../scripts/e-bend.json"),
        marks: include_str!("../scripts/e-bend.marks.json"),
        kinds: &[Analysis::Frequency],
        group: Group::Junctions,
    },
    Template {
        name: "h-tee",
        title: "H-plane tee",
        label: "H-plane T-junction in WR-90: a side arm off the middle of a straight guide, three ports",
        script: include_str!("../scripts/h-tee.json"),
        marks: include_str!("../scripts/h-tee.marks.json"),
        kinds: &[Analysis::Frequency],
        group: Group::Junctions,
    },
    Template {
        name: "h-bend-radius",
        title: "H-plane radius bend",
        label: "H-plane 90° radius bend in WR-90: inner radius r (28.6 mm: a mean radius of one guide wavelength at 10 GHz), each arc drawn in 18 straight sides; ports on the arms' ends across y and x",
        script: include_str!("../scripts/h-bend-radius.json"),
        marks: include_str!("../scripts/h-bend-radius.marks.json"),
        kinds: &[Analysis::Frequency],
        group: Group::Junctions,
    },
    Template {
        name: "e-bend-radius",
        title: "E-plane radius bend",
        label: "E-plane 90° radius bend in WR-90: inner radius r (35 mm: a mean radius of one guide wavelength at 10 GHz), each arc drawn in 18 straight sides; ports on the arms' ends across y and x",
        script: include_str!("../scripts/e-bend-radius.json"),
        marks: include_str!("../scripts/e-bend-radius.marks.json"),
        kinds: &[Analysis::Frequency],
        group: Group::Junctions,
    },
    Template {
        name: "e-tee",
        title: "E-plane tee",
        label: "E-plane T-junction in WR-90: a side arm off the broad wall of a straight guide, three ports; from the side arm the two ends are fed equally in antiphase",
        script: include_str!("../scripts/e-tee.json"),
        marks: include_str!("../scripts/e-tee.marks.json"),
        kinds: &[Analysis::Frequency],
        group: Group::Junctions,
    },
    Template {
        name: "magic-tee",
        title: "Magic tee",
        label: "Magic tee (hybrid T) in WR-90: collinear arms 1 and 2, the H arm 3 and the E arm 4, each matched by posts at 9.5 GHz (two in the H arm, one across the E arm); 3 and 4 isolated, each split equally between 1 and 2, in phase from the H arm and in antiphase from the E arm",
        script: include_str!("../scripts/magic-tee.json"),
        marks: include_str!("../scripts/magic-tee.marks.json"),
        kinds: &[Analysis::Frequency],
        group: Group::Junctions,
    },
    Template {
        name: "inductive-post",
        title: "Inductive post",
        label: "A 3 mm conducting post across WR-90, wall to wall, cut from the box by a Boolean difference: a strong shunt inductance",
        script: include_str!("../scripts/inductive-post.json"),
        marks: include_str!("../scripts/inductive-post.marks.json"),
        kinds: &[Analysis::Frequency],
        group: Group::Junctions,
    },
    Template {
        name: "td-eplane-step",
        title: "E-plane step",
        label: "E-plane step in WR-90 from full to half height (10.16 to 5.08 mm), in the time domain",
        script: include_str!("../scripts/td-eplane-step.json"),
        marks: "{}",
        kinds: &[Analysis::TimeDomain],
        group: Group::Junctions,
    },
    Template {
        name: "td-h-bend",
        title: "H-plane bend",
        label: "Mitred H-plane bend of WR-90 in the time domain, the solid staircased on 1 mm cells, both ports excited: |S11| from −40 to −19 dB across the band",
        script: include_str!("../scripts/td-h-bend.json"),
        marks: "{}",
        kinds: &[Analysis::TimeDomain],
        group: Group::Junctions,
    },
    Template {
        name: "td-e-bend",
        title: "E-plane bend",
        label: "Mitred E-plane bend of WR-90 in the time domain, both ports excited: |S11| under −31 dB across the band",
        script: include_str!("../scripts/td-e-bend.json"),
        marks: "{}",
        kinds: &[Analysis::TimeDomain],
        group: Group::Junctions,
    },
    Template {
        name: "td-h-bend-radius",
        title: "H-plane radius bend",
        label: "Curved H-plane bend of WR-90 in the time domain, its arc staircased: |S11| under −33 dB",
        script: include_str!("../scripts/td-h-bend-radius.json"),
        marks: "{}",
        kinds: &[Analysis::TimeDomain],
        group: Group::Junctions,
    },
    Template {
        name: "td-e-bend-radius",
        title: "E-plane radius bend",
        label: "Curved E-plane bend of WR-90 in the time domain, its arc staircased: |S11| under −26 dB",
        script: include_str!("../scripts/td-e-bend-radius.json"),
        marks: "{}",
        kinds: &[Analysis::TimeDomain],
        group: Group::Junctions,
    },
    Template {
        name: "td-inductive-post",
        title: "Inductive post",
        label: "A 3 mm post across WR-90 in the time domain on 0.75 mm cells: the shunt inductance's |S11| of about 0.96 at 10 GHz (the staircase makes the post a little thicker than the exact 0.953)",
        script: include_str!("../scripts/td-inductive-post.json"),
        marks: "{}",
        kinds: &[Analysis::TimeDomain],
        group: Group::Junctions,
    },
    Template {
        name: "td-h-tee",
        title: "H-plane tee",
        label: "H-plane tee of WR-90 in the time domain, every port excited: the full 3 × 3 S-matrix, the side arms in phase",
        script: include_str!("../scripts/td-h-tee.json"),
        marks: "{}",
        kinds: &[Analysis::TimeDomain],
        group: Group::Junctions,
    },
    Template {
        name: "td-e-tee",
        title: "E-plane tee",
        label: "E-plane tee of WR-90 in the time domain, every port excited: the side arms in antiphase",
        script: include_str!("../scripts/td-e-tee.json"),
        marks: "{}",
        kinds: &[Analysis::TimeDomain],
        group: Group::Junctions,
    },
    Template {
        name: "td-magic-tee",
        title: "Magic tee",
        label: "Magic tee in the time domain on 1.5 mm cells, port 4 excited: isolated from port 3 (−59 dB or less), splitting equally into ports 1 and 2",
        script: include_str!("../scripts/td-magic-tee.json"),
        marks: "{}",
        kinds: &[Analysis::TimeDomain],
        group: Group::Junctions,
    },
    Template {
        name: "pcb-bend",
        title: "Mitred microstrip bend",
        label: "Microstrip right-angle bend, its corner mitred by a triangle of copper: |S11| about −19 dB up to 5 GHz",
        script: include_str!("../scripts/pcb-bend.json"),
        marks: "{}",
        kinds: &[Analysis::Frequency],
        group: Group::Junctions,
    },
    // Couplers.
    Template {
        name: "cross-guide-coupler",
        title: "Cross-guide coupler",
        label: "Cross-guide directional coupler: two WR-90 guides crossed at right angles over a common broad wall t thick, coupled through two round holes on the diagonal of their overlap; four ports, about 30 dB coupled",
        script: include_str!("../scripts/cross-guide-coupler.json"),
        marks: include_str!("../scripts/cross-guide-coupler.marks.json"),
        kinds: &[Analysis::Frequency],
        group: Group::Couplers,
    },
    Template {
        name: "td-cross-guide-coupler",
        title: "Cross-guide coupler",
        label: "Cross-guide coupler in the time domain: two WR-90 guides crossing over coupling slots, port 1 excited, the coupled and isolated ports",
        script: include_str!("../scripts/td-cross-guide-coupler.json"),
        marks: "{}",
        kinds: &[Analysis::TimeDomain],
        group: Group::Couplers,
    },
    Template {
        name: "pcb-branch-line",
        title: "Branch-line coupler",
        label: "Microstrip branch-line coupler: 35 Ω and 50 Ω quarter-wave arms in a square, port 1 split equally to 2 and 3 in quadrature near 3.2 GHz, 4 isolated",
        script: include_str!("../scripts/pcb-branch-line.json"),
        marks: "{}",
        kinds: &[Analysis::Frequency],
        group: Group::Couplers,
    },
    Template {
        name: "pcb-coupled-line",
        title: "Coupled-line coupler",
        label: "Microstrip coupled-line coupler: two 50 Ω lines 0.2 mm apart over 9 mm, coupled backwards by about 12.5 dB at 4.8 GHz; four ports",
        script: include_str!("../scripts/pcb-coupled-line.json"),
        marks: "{}",
        kinds: &[Analysis::Frequency],
        group: Group::Couplers,
    },
    Template {
        name: "pcb-wilkinson",
        title: "Wilkinson divider",
        label: "Wilkinson power divider in microstrip: two 70.7 Ω quarter-wave arms and a 100 Ω chip resistor (pcb.element) across the outputs: an equal split at −3.0 dB, the outputs isolated by 25 dB at 2 GHz",
        script: include_str!("../scripts/pcb-wilkinson.json"),
        marks: "{}",
        kinds: &[Analysis::Frequency],
        group: Group::Couplers,
    },
    // Materials.
    Template {
        name: "ferrite-phase-shifter",
        title: "Ferrite phase shifter",
        label: "Ferrite phase shifter: a 2 mm YIG slab (4πMs = 1780 G, εr = 13) off the middle of WR-90, biased across the height at 1000 Oe, 20 mm long: non-reciprocal — the wave going one way lags the other by tens of degrees, |S21| = |S12| (solved by LU, a frequency at a time)",
        script: include_str!("../scripts/ferrite-phase-shifter.json"),
        marks: "{}",
        kinds: &[Analysis::Frequency],
        group: Group::Materials,
    },
    Template {
        name: "material-sample",
        title: "Material measurement",
        label: "Material measurement: a sample (εr = 4, tan δ = 0.01) 8 mm long filling WR-90 between 10 mm of empty guide each side, its S over X band; after Run, Material from S (the Analyses view) reads εr and μr back by Nicolson–Ross–Weir",
        script: include_str!("../scripts/material-sample.json"),
        marks: "{}",
        kinds: &[Analysis::Frequency],
        group: Group::Materials,
    },
    Template {
        name: "termination",
        title: "Matched load",
        label: "Matched load: a lossy wedge (εr = 1.5, tan δ = 0.5) 90 mm long, 1 mm clear of the walls, against the short at the end of a WR-90 guide; one port: |S11| below 0.06 over the band",
        script: include_str!("../scripts/termination.json"),
        marks: include_str!("../scripts/termination.marks.json"),
        kinds: &[Analysis::Frequency],
        group: Group::Materials,
    },
    Template {
        name: "td-dielectric-slab",
        title: "Dielectric slab",
        label: "A slab of dielectric (εr = 2.2, 20 mm thick) across WR-90 in the time domain: its Fabry–Pérot reflection over the band, matched where the slab is half a guide wavelength",
        script: include_str!("../scripts/td-dielectric-slab.json"),
        marks: "{}",
        kinds: &[Analysis::TimeDomain],
        group: Group::Materials,
    },
    Template {
        name: "td-lossy-slab",
        title: "Lossy slab",
        label: "A lossy slab (εr = 2.2, tan δ = 0.05) across WR-90 in the time domain, its loss a conductivity matched at the band's centre",
        script: include_str!("../scripts/td-lossy-slab.json"),
        marks: "{}",
        kinds: &[Analysis::TimeDomain],
        group: Group::Materials,
    },
    Template {
        name: "td-dispersive-slab",
        title: "Dispersive slab",
        label: "A dispersive slab across WR-90 (εr = 2 with a Debye and a Lorentz pole) in the time domain, its poles by auxiliary differential equations",
        script: include_str!("../scripts/td-dispersive-slab.json"),
        marks: "{}",
        kinds: &[Analysis::TimeDomain],
        group: Group::Materials,
    },
    Template {
        name: "td-termination",
        title: "Lossy wedge load",
        label: "A lossy wedge closing WR-90 in the time domain: |S11| under −25 dB across the band, the loss tangent as a conductivity matched at the band's centre",
        script: include_str!("../scripts/td-termination.json"),
        marks: "{}",
        kinds: &[Analysis::TimeDomain],
        group: Group::Materials,
    },
    // Periodic.
    Template {
        name: "corrugated-guide",
        title: "Corrugated waveguide",
        label: "WR-90 whose narrow wall waves, b + h sin(2πz/p) for n periods, a SolidCraft design with the wave an equation curve: the corrugation slows the wave, S21's phase lagging a straight guide's (23° at 10 GHz for h = 2 mm, p = 10 mm, n = 4)",
        script: include_str!("../scripts/corrugated-guide.json"),
        marks: include_str!("../scripts/corrugated-guide.marks.json"),
        kinds: &[Analysis::Frequency],
        group: Group::Periodic,
    },
    Template {
        name: "pulse-compressor",
        title: "Pulse compressor",
        label: "Pulse compressor: one 16 mm cell of a circular guide (radius 12 mm) whose wall waves about its axis, r0 + h sin(2πz/p), a SolidCraft design revolved; its TE11 Bloch wave on the half-plane (solve.revolution.bloch) and a 20 ns chirped pulse through 3 m of it, the chirp matched to the group delay: compressed about ×8, its tail cut",
        script: include_str!("../scripts/pulse-compressor.json"),
        marks: "{}",
        kinds: &[Analysis::Frequency],
        group: Group::Periodic,
    },
    Template {
        name: "helical-compressor",
        title: "Helical pulse compressor",
        label: "Helical pulse compressor: one 36 mm period of a circular guide (radius 14 mm) whose wall turns as a three-lobed helix, r0 + h cos(3φ − 2πz/p), a SolidCraft sweep with a twist; its Bloch waves solved in 3D (solve.bloch on Run, about a minute) — the helix couples one rotation of TE11 with TE21 near its cut-off, a wave whose delay rises steeply with frequency — and a 12 ns chirped pulse through 3 m of it, the chirp matched: compressed about ×10 to 0.8 ns with no tail to speak of",
        script: include_str!("../scripts/helical-compressor.json"),
        marks: "{}",
        kinds: &[Analysis::Frequency],
        group: Group::Periodic,
    },
    // Resonators.
    Template {
        name: "cavity",
        title: "Rectangular cavity",
        label: "Copper rectangular cavity, 22.86 × 10.16 × 30 mm: its four lowest resonances (TE101 at 8.24 GHz) and Q factors",
        script: include_str!("../scripts/cavity.json"),
        marks: include_str!("../scripts/cavity.marks.json"),
        kinds: &[Analysis::Eigenmodes],
        group: Group::Resonators,
    },
    Template {
        name: "cylinder-cavity",
        title: "Cylindrical cavity",
        label: "Copper cylindrical cavity, radius 10 mm, 30 mm long, meshed with unstructured tetrahedra: TE111 and TM010 against the Bessel zeros",
        script: include_str!("../scripts/cylinder-cavity.json"),
        marks: include_str!("../scripts/cylinder-cavity.marks.json"),
        kinds: &[Analysis::Eigenmodes],
        group: Group::Resonators,
    },
    Template {
        name: "electron-pillbox",
        title: "Electrons in a pillbox",
        label: "Electrons through a copper pillbox 10 mm in radius and 13.06 mm long: its TM010 at 1 MV, 1 MeV electrons sent at twelve phases and five distances off the axis, their energy gained against qV₀T cos φ",
        script: include_str!("../scripts/electron-pillbox.json"),
        marks: include_str!("../scripts/electron-pillbox.marks.json"),
        kinds: &[Analysis::Eigenmodes],
        group: Group::Resonators,
    },
    Template {
        name: "loaded-sphere",
        title: "Loaded spherical cavity",
        label: "Spherical cavity, radius 10 mm, around a concentric ceramic sphere (εr = 4, radius 4 mm): two material regions joined node for node, the TM11 triplet at 11.26 GHz",
        script: include_str!("../scripts/loaded-sphere.json"),
        marks: include_str!("../scripts/loaded-sphere.marks.json"),
        kinds: &[Analysis::Eigenmodes],
        group: Group::Resonators,
    },
    Template {
        name: "sphere-cavity",
        title: "Spherical cavity",
        label: "Copper spherical cavity, radius 10 mm: the TM11 triplet at 13.09 GHz (the first root of the spherical Bessel function's derivative) and its Q",
        script: include_str!("../scripts/sphere-cavity.json"),
        marks: include_str!("../scripts/sphere-cavity.marks.json"),
        kinds: &[Analysis::Eigenmodes],
        group: Group::Resonators,
    },
    Template {
        name: "coaxial-cavity",
        title: "Coaxial resonator",
        label: "Coaxial resonator shorted at both ends, conductors of radius 1 and 3.5 mm, 30 mm long: TEM half-wave resonances at multiples of c/2l (5 GHz) and their copper Q",
        script: include_str!("../scripts/coaxial-cavity.json"),
        marks: include_str!("../scripts/coaxial-cavity.marks.json"),
        kinds: &[Analysis::Eigenmodes],
        group: Group::Resonators,
    },
    Template {
        name: "slab-cavity",
        title: "Partly filled cavity",
        label: "Rectangular cavity filled in part along its length: WR-90 with 15 mm of vacuum and 15 mm of dielectric (εr = 2.2), its TE10p resonances those of the two sections' transcendental equation",
        script: include_str!("../scripts/slab-cavity.json"),
        marks: include_str!("../scripts/slab-cavity.marks.json"),
        kinds: &[Analysis::Eigenmodes],
        group: Group::Resonators,
    },
    Template {
        name: "dielectric-resonator",
        title: "Dielectric resonator",
        label: "Dielectric resonator: a ceramic puck (εr = 38, radius 5.25 mm, 4.6 mm high) at the centre of a copper box 30 × 30 × 20 mm, its TE01δ resonance near 4.96 GHz and its Q",
        script: include_str!("../scripts/dielectric-resonator.json"),
        marks: include_str!("../scripts/dielectric-resonator.marks.json"),
        kinds: &[Analysis::Eigenmodes],
        group: Group::Resonators,
    },
    Template {
        name: "combline-resonator",
        title: "Combline resonator",
        label: "Combline resonator: a post 15 mm high on the floor of a 20 mm copper box, a quarter-wave line loaded by the lid at its open end, 4.0 GHz",
        script: include_str!("../scripts/combline-resonator.json"),
        marks: include_str!("../scripts/combline-resonator.marks.json"),
        kinds: &[Analysis::Eigenmodes],
        group: Group::Resonators,
    },
    Template {
        name: "reentrant-cavity",
        title: "Re-entrant cavity",
        label: "Re-entrant cavity: a post along the axis of a copper cylinder (radius 15 mm, 20 mm long) stopping 2 mm short of its lid, the gap's capacitance bringing the resonance down to 2.75 GHz",
        script: include_str!("../scripts/reentrant-cavity.json"),
        marks: include_str!("../scripts/reentrant-cavity.marks.json"),
        kinds: &[Analysis::Eigenmodes],
        group: Group::Resonators,
    },
    Template {
        name: "pcb-ring",
        title: "Ring resonator",
        label: "Square ring resonator in microstrip, gap-coupled at both ends on a lossy substrate (tan δ 0.0027): its transmission peaks at 2.0 and 4.0 GHz, where the ring is one and two wavelengths round",
        script: include_str!("../scripts/pcb-ring.json"),
        marks: "{}",
        kinds: &[Analysis::Frequency],
        group: Group::Resonators,
    },
    // Dispersion.
    Template {
        name: "dispersion-guide",
        title: "Empty guide",
        label: "Dispersion diagram of an empty WR-90 guide taken as a 10 mm period: TE10's Bloch band from its 6.56 GHz cut-off to 16.4 GHz at the zone's edge, against the closed form",
        script: include_str!("../scripts/dispersion-guide.json"),
        marks: "{}",
        kinds: &[Analysis::Eigenmodes],
        group: Group::Dispersion,
    },
    Template {
        name: "dispersion-loaded",
        title: "Slab-loaded guide",
        label: "A guide loaded periodically with dielectric slabs (εr 4, 4 mm in every 10 mm): its Bloch bands and the stop band between them at the zone's edge, against the closed form of TE10's",
        script: include_str!("../scripts/dispersion-loaded.json"),
        marks: "{}",
        kinds: &[Analysis::Eigenmodes],
        group: Group::Dispersion,
    },
    Template {
        name: "dispersion-iris",
        title: "Iris-loaded guide",
        label: "An iris-loaded guide (an inductive iris 11 mm wide every 12 mm): the passband the irises leave, from 10.4 to 14.7 GHz across the zone",
        script: include_str!("../scripts/dispersion-iris.json"),
        marks: "{}",
        kinds: &[Analysis::Eigenmodes],
        group: Group::Dispersion,
    },
    Template {
        name: "dispersion-corrugated",
        title: "Corrugated guide",
        label: "A corrugated guide (E-plane teeth leaving a 4 mm gap, 8 mm period): a slow-wave band flattening to 10.9 GHz at the zone's edge",
        script: include_str!("../scripts/dispersion-corrugated.json"),
        marks: "{}",
        kinds: &[Analysis::Eigenmodes],
        group: Group::Dispersion,
    },
    // Antennas.
    Template {
        name: "dipole",
        title: "Short dipole",
        label: "A 1 mm current element at 10 GHz in a box lined with a perfectly matched layer: radiated power and directivity against the Hertzian dipole",
        script: include_str!("../scripts/dipole.json"),
        marks: "{}",
        kinds: &[Analysis::Radiation],
        group: Group::Antennas,
    },
    Template {
        name: "halfwave-dipole",
        title: "Half-wave dipole",
        label: "A half-wave wire dipole at 10 GHz (radius 0.5 mm, 1 mm gap) fed by a lumped port: input impedance, S11, power balance and pattern",
        script: include_str!("../scripts/halfwave-dipole.json"),
        marks: include_str!("../scripts/halfwave-dipole.marks.json"),
        kinds: &[Analysis::Radiation],
        group: Group::Antennas,
    },
    Template {
        name: "open-waveguide",
        title: "Open-ended waveguide",
        label: "An open-ended WR-90 channel in a metal block radiating at 10 GHz, fed by a waveguide port on the channel's closed end: S11, gain and pattern",
        script: include_str!("../scripts/open-waveguide.json"),
        marks: "{}",
        kinds: &[Analysis::Radiation],
        group: Group::Antennas,
    },
    Template {
        name: "horn",
        title: "Pyramidal horn",
        label: "Pyramidal horn fed by WR-90, a SolidCraft design: an A × B mouth h past a feed f long, walls t thick, radiating at 10 GHz in an open box: S11, gain and pattern",
        script: include_str!("../scripts/horn.json"),
        marks: include_str!("../scripts/horn.marks.json"),
        kinds: &[Analysis::Radiation],
        group: Group::Antennas,
    },
    Template {
        name: "yagi",
        title: "Yagi–Uda antenna",
        label: "Three-element Yagi–Uda antenna at 10 GHz: a wire dipole fed by a lumped port between a longer reflector and a shorter director, about 7.5 dBi towards the director",
        script: include_str!("../scripts/yagi.json"),
        marks: "{}",
        kinds: &[Analysis::Radiation],
        group: Group::Antennas,
    },
    Template {
        name: "monopole",
        title: "Quarter-wave monopole",
        label: "Quarter-wave monopole on a 40 mm square ground plate at 10 GHz, fed by a lumped port at its base: input impedance, S11 and pattern",
        script: include_str!("../scripts/monopole.json"),
        marks: "{}",
        kinds: &[Analysis::Radiation],
        group: Group::Antennas,
    },
    Template {
        name: "heart-loop",
        title: "Heart-shaped loop",
        label: "Heart-shaped wire loop at 2.45 GHz, a SolidCraft design: a 0.5 mm wire bent into a heart W wide (a loop about a wavelength round), fed across a 1 mm gap at its tip by a 50 Ω lumped port: input impedance, gain, polarization and pattern",
        script: include_str!("../scripts/heart-loop.json"),
        marks: include_str!("../scripts/heart-loop.marks.json"),
        kinds: &[Analysis::Radiation],
        group: Group::Antennas,
    },
    Template {
        name: "patch-antenna",
        title: "Patch antenna",
        label: "Inset-fed microstrip patch antenna at 10 GHz on 0.787 mm of εr = 2.2 over a finite ground, a board in an open region: S11 about −17 dB, about 7 dBi broadside",
        script: include_str!("../scripts/patch-antenna.json"),
        marks: "{}",
        kinds: &[Analysis::Radiation, Analysis::Pcb],
        group: Group::Antennas,
    },
    Template {
        name: "adsb-patch",
        title: "ADS-B patch",
        label: "ADS-B receiving patch at 1090 MHz on 1.6 mm of FR-4 (εr = 4.4, tan δ = 0.02), inset-fed by a 50 Ω line over a finite ground, a board in an open region: S11 about −20 dB, 6.4 dBi directivity broadside, 1.2 dBi gain (the FR-4's loss: efficiency 0.3)",
        script: include_str!("../scripts/adsb-patch.json"),
        marks: "{}",
        kinds: &[Analysis::Radiation, Analysis::Pcb],
        group: Group::Antennas,
    },
    Template {
        name: "td-dipole",
        title: "Dipole",
        label: "A centre-fed dipole of square arms in the time domain, its 1 mm gap carried to the arms on the grid: resonance near 10 GHz, S11 against 50 Ω and the radiated power",
        script: include_str!("../scripts/td-dipole.json"),
        marks: "{}",
        kinds: &[Analysis::TimeDomain, Analysis::Radiation],
        group: Group::Antennas,
    },
    // Scattering.
    Template {
        name: "sphere-scattering",
        title: "Ceramic sphere",
        label: "A plane wave on a ceramic sphere (εr = 4, radius 5 mm) at 10 GHz: the scattered field and the radar cross-section against the Mie series",
        script: include_str!("../scripts/sphere-scattering.json"),
        marks: include_str!("../scripts/sphere-scattering.marks.json"),
        kinds: &[Analysis::Radiation],
        group: Group::Scattering,
    },
    Template {
        name: "metal-sphere-scattering",
        title: "Metal sphere",
        label: "A plane wave on a metal sphere (radius 5 mm) at 10 GHz: the scattered field and the scattering cross-section against the Mie series",
        script: include_str!("../scripts/metal-sphere-scattering.json"),
        marks: "{}",
        kinds: &[Analysis::Radiation],
        group: Group::Scattering,
    },
    Template {
        name: "td-sphere-scattering",
        title: "Ceramic sphere",
        label: "A plane wave on a ceramic sphere (εr 4, radius 5 mm) in the time domain, by a total-field/scattered-field split: its scattering cross-section over 6–14 GHz in one run, against the Mie series (about 10 % high on 0.5 mm cells)",
        script: include_str!("../scripts/td-sphere-scattering.json"),
        marks: "{}",
        kinds: &[Analysis::TimeDomain, Analysis::Radiation],
        group: Group::Scattering,
    },
    Template {
        name: "td-metal-sphere-scattering",
        title: "Metal sphere",
        label: "A plane wave on a metal sphere of radius 5 mm in the time domain: its scattering and radar cross-sections over 6–14 GHz in one run, against the Mie series (the staircased sphere scatters a little more)",
        script: include_str!("../scripts/td-metal-sphere-scattering.json"),
        marks: "{}",
        kinds: &[Analysis::TimeDomain, Analysis::Radiation],
        group: Group::Scattering,
    },
    // Boards.
    Template {
        name: "pcb-microstrip",
        title: "Microstrip over GND",
        label: "PCB microstrip over continuous GND: impedance, delay, S-parameters and TDR estimates",
        script: include_str!("../scripts/pcb-microstrip.json"),
        marks: "{}",
        kinds: &[Analysis::Pcb],
        group: Group::Boards,
    },
    Template {
        name: "pcb-split-ground",
        title: "Trace over a split GND",
        label: "PCB trace crossing a ground-plane slot: locate the broken return reference",
        script: include_str!("../scripts/pcb-split-ground.json"),
        marks: "{}",
        kinds: &[Analysis::Pcb],
        group: Group::Boards,
    },
    Template {
        name: "pcb-parallel",
        title: "Parallel traces",
        label: "PCB parallel signal routes: screen nearby traces and inspect the reference plane",
        script: include_str!("../scripts/pcb-parallel.json"),
        marks: "{}",
        kinds: &[Analysis::Pcb],
        group: Group::Boards,
    },
    Template {
        name: "pcb-board",
        title: "Four-layer board",
        label: "Four-layer board: a route through a via, a differential pair, a clock over a slotted ground",
        script: include_str!("../scripts/pcb-board.json"),
        marks: "{}",
        kinds: &[Analysis::Pcb],
        group: Group::Boards,
    },
    // Systems.
    Template {
        name: "adsb-receiver",
        title: "ADS-B receiver chain",
        label: "ADS-B receiver chain at 1090 MHz: the adsb-patch antenna's terminals, an LNA (20 dB, 0.6 dB, OIP3 30 dBm), the adsb-bandpass filter and a receiver (20 dB, 6 dB), the antenna's and filter's S as EMCraft solved them: from the antenna 0.84 dB noise figure and 37.4 dB gain at 1.09 GHz, −100 dBm sensitivity in 2 MHz at 10 dB SNR, the budget stage by stage",
        script: include_str!("../scripts/adsb-receiver.json"),
        marks: "{}",
        kinds: &[Analysis::System],
        group: Group::Systems,
    },
    Template {
        name: "planar-diode",
        title: "Planar diode",
        label: "A planar diode by particle-in-cell: a cathode emitting electrons at rest as the field allows, an anode 10 mm away held 100 kV above it; the current settles near the space-charge limit, Jory & Trivelpiece's 7.2e5 A/m² (relativistic Child–Langmuir)",
        script: include_str!("../scripts/planar-diode.json"),
        marks: "{}",
        kinds: &[Analysis::Particles],
        group: Group::Particles,
    },
    Template {
        name: "pillbox-wake",
        title: "Bunch through a pillbox",
        label: "A bunch through a closed pillbox by particle-in-cell: 1 nC, σ_z = 2 mm, 10 GeV electrons on the axis of a perfectly conducting cavity 10 mm in radius and 13.06 mm long; the energy it leaves, its loss factor, 5.8 V/pC over the TM0np modes",
        script: include_str!("../scripts/pillbox-wake.json"),
        marks: "{}",
        kinds: &[Analysis::Particles],
        group: Group::Particles,
    },
    Template {
        name: "cherenkov-pillbox",
        title: "Cherenkov pillbox",
        label: "Cherenkov light in a dielectric-filled pillbox by particle-in-cell: a 1 pC bunch at 10 GeV along a perfectly conducting cavity 10 mm in radius and 39.18 mm long filled with εr = 2.25, faster than light there (β > 1/n); its loss, 25.7 V/pC, grows with the length",
        script: include_str!("../scripts/cherenkov-pillbox.json"),
        marks: "{}",
        kinds: &[Analysis::Particles],
        group: Group::Particles,
    },
    Template {
        name: "drift-tube",
        title: "Ring beam in a drift tube",
        label: "A ring beam in a drift tube by particle-in-cell: 750 A of 128 keV electrons on a 6 mm ring in a grounded tube 10 mm in radius, held by 10 T along it; its space charge slows it to γ 1.162, below the limiting current of 1.07 kA (Bogdankevich & Rukhadze)",
        script: include_str!("../scripts/drift-tube.json"),
        marks: "{}",
        kinds: &[Analysis::Particles],
        group: Group::Particles,
    },
    Template {
        name: "relativistic-bwo",
        title: "Relativistic BWO",
        label: "A relativistic backward-wave oscillator by particle-in-cell in a body of revolution, the Ku-band one of Na et al. (2017): 1.5 kA of 553 keV electrons on a 9–11 mm ring held by 5 T through 14.5 ripples of 15 ± 2.1 mm every 7.5 mm, open at the collector; it oscillates near 15 GHz on these 0.5 mm cells (15.59 GHz published, its walls smooth)",
        script: include_str!("../scripts/relativistic-bwo.json"),
        marks: "{}",
        kinds: &[Analysis::Particles],
        group: Group::Particles,
    },
    Template {
        name: "two-stream",
        title: "Two-stream instability",
        label: "The two-stream instability by particle-in-cell: two cold electron streams of 10¹⁶ m⁻³ at ±3·10⁷ m/s in a periodic box one wavelength long; the wave between them grows at ω_b/2 = 2.8·10⁹ /s until they trap it",
        script: include_str!("../scripts/two-stream.json"),
        marks: "{}",
        kinds: &[Analysis::Particles],
        group: Group::Particles,
    },
    Template {
        name: "plasma-oscillation",
        title: "Plasma oscillation",
        label: "A cold plasma by particle-in-cell: 10¹⁶ electrons per m³, periodic along every axis, a velocity wave along it; its field rings at the plasma frequency ω_p/2π = 0.898 GHz",
        script: include_str!("../scripts/plasma-oscillation.json"),
        marks: "{}",
        kinds: &[Analysis::Particles],
        group: Group::Particles,
    },
];

/// A file a template is built from, compiled in: no file is read when a template opens it.
pub struct Design {
    /// Its file name: `<name>.sc`, a SolidCraft design; `<name>.s<n>p`, S-parameters.
    pub file: &'static str,
    pub bytes: &'static [u8],
}

/// How a script names a template's file: `template:<file>` (`geometry.import`'s and
/// `results.import`'s `path`).
pub const DESIGN_SCHEME: &str = "template:";

/// The files templates are built from: SolidCraft designs, each built by SolidCraft's command
/// line from the command script beside it (`designs/<name>.script.json`, `cargo xtask designs`), and Touchstone
/// files of templates' S as EMCraft solves them (`adsb-receiver`'s antenna and filter; each
/// file's comment names the template and the solve that wrote it).
pub const DESIGNS: &[Design] = &[
    Design { file: "taper.sc", bytes: include_bytes!("../designs/taper.sc") },
    Design { file: "horn.sc", bytes: include_bytes!("../designs/horn.sc") },
    Design { file: "corrugated-h.sc", bytes: include_bytes!("../designs/corrugated-h.sc") },
    Design { file: "corrugated-e.sc", bytes: include_bytes!("../designs/corrugated-e.sc") },
    Design { file: "corrugated-cell.sc", bytes: include_bytes!("../designs/corrugated-cell.sc") },
    Design { file: "helical-cell.sc", bytes: include_bytes!("../designs/helical-cell.sc") },
    Design { file: "heart-wire.sc", bytes: include_bytes!("../designs/heart-wire.sc") },
    Design { file: "h-bend.sc", bytes: include_bytes!("../designs/h-bend.sc") },
    Design { file: "cylinder-cavity.sc", bytes: include_bytes!("../designs/cylinder-cavity.sc") },
    Design { file: "coaxial-line.sc", bytes: include_bytes!("../designs/coaxial-line.sc") },
    Design { file: "inductive-post.sc", bytes: include_bytes!("../designs/inductive-post.sc") },
    Design { file: "metal-sphere.sc", bytes: include_bytes!("../designs/metal-sphere.sc") },
    Design { file: "cross-guide-coupler.sc", bytes: include_bytes!("../designs/cross-guide-coupler.sc") },
    Design { file: "straight.sc", bytes: include_bytes!("../designs/straight.sc") },
    Design { file: "circular-guide.sc", bytes: include_bytes!("../designs/circular-guide.sc") },
    Design { file: "reentrant-cavity.sc", bytes: include_bytes!("../designs/reentrant-cavity.sc") },
    Design { file: "combline-resonator.sc", bytes: include_bytes!("../designs/combline-resonator.sc") },
    Design { file: "dielectric-resonator.sc", bytes: include_bytes!("../designs/dielectric-resonator.sc") },
    Design { file: "ferrite-phase-shifter.sc", bytes: include_bytes!("../designs/ferrite-phase-shifter.sc") },
    Design { file: "sphere-cavity.sc", bytes: include_bytes!("../designs/sphere-cavity.sc") },
    Design { file: "loaded-sphere.sc", bytes: include_bytes!("../designs/loaded-sphere.sc") },
    Design { file: "termination.sc", bytes: include_bytes!("../designs/termination.sc") },
    Design { file: "h-tee.sc", bytes: include_bytes!("../designs/h-tee.sc") },
    Design { file: "e-tee.sc", bytes: include_bytes!("../designs/e-tee.sc") },
    Design { file: "e-bend.sc", bytes: include_bytes!("../designs/e-bend.sc") },
    Design { file: "h-bend-radius.sc", bytes: include_bytes!("../designs/h-bend-radius.sc") },
    Design { file: "e-bend-radius.sc", bytes: include_bytes!("../designs/e-bend-radius.sc") },
    Design { file: "magic-tee.sc", bytes: include_bytes!("../designs/magic-tee.sc") },
    Design { file: "sphere-scattering.sc", bytes: include_bytes!("../designs/sphere-scattering.sc") },
    Design { file: "halfwave-dipole.sc", bytes: include_bytes!("../designs/halfwave-dipole.sc") },
    Design { file: "monopole.sc", bytes: include_bytes!("../designs/monopole.sc") },
    Design { file: "yagi.sc", bytes: include_bytes!("../designs/yagi.sc") },
    Design { file: "open-waveguide.sc", bytes: include_bytes!("../designs/open-waveguide.sc") },
    Design { file: "adsb-patch.s1p", bytes: include_bytes!("../designs/adsb-patch.s1p") },
    Design { file: "adsb-bandpass.s2p", bytes: include_bytes!("../designs/adsb-bandpass.s2p") },
];

/// The design a path names, `template:<file>`; `None` for any other path (a file of the
/// user's) or a design no template has.
///
/// ```
/// let taper = emcraft_templates::design("template:taper.sc");
/// assert!(taper.is_some_and(|b| !b.is_empty()));
/// assert!(emcraft_templates::design("taper.sc").is_none());
/// assert!(emcraft_templates::design("template:nothing.sc").is_none());
/// ```
pub fn design(path: &str) -> Option<&'static [u8]> {
    let file = path.strip_prefix(DESIGN_SCHEME)?;
    DESIGNS.iter().find(|d| d.file == file).map(|d| d.bytes)
}

/// The template called `name`.
pub fn find(name: &str) -> Option<&'static Template> {
    TEMPLATES.iter().find(|t| t.name == name)
}

/// The templates made for `kind`, in [`TEMPLATES`]' order.
///
/// ```
/// use emcraft_templates::{Analysis, made_for};
/// assert!(made_for(Analysis::Eigenmodes).any(|t| t.name == "cavity"));
/// assert!(!made_for(Analysis::Eigenmodes).any(|t| t.name == "dipole"));
/// ```
pub fn made_for(kind: Analysis) -> impl Iterator<Item = &'static Template> {
    TEMPLATES.iter().filter(move |t| t.kinds.contains(&kind))
}

/// The templates' names, in [`TEMPLATES`]' order.
pub fn names() -> Vec<&'static str> {
    TEMPLATES.iter().map(|t| t.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_are_listed_group_by_group_and_named_once() {
        let groups: Vec<Group> = TEMPLATES.iter().map(|t| t.group).collect();
        assert!(groups.is_sorted(), "keep TEMPLATES in Group::ALL's order");
        let names = names();
        let mut unique = names.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), names.len());
        assert!(Group::ALL.is_sorted() && Group::ALL.iter().all(|g| groups.contains(g)), "every group has a template");
    }
}
