//! Particle-in-cell runs by command (docs/adr/0012-particles.md, stage B): the sources set,
//! checked and undone; `solve.particles` against a cold plasma's ω_p and a planar diode's
//! space-charge-limited current; a run saved and opened again.

use serde_json::{Value, json};

use crate::Session;
use crate::tests::scratch;

fn run(s: &mut Session, steps: &[(&str, Value)]) {
    for (c, p) in steps {
        s.execute(c, p).unwrap_or_else(|e| panic!("{c}: {e}"));
    }
}

/// A cold plasma of 10¹⁶ m⁻³ along a periodic box of 32 cells, its velocities a wave of one
/// wavelength, a probe a quarter of the way along.
fn plasma() -> Session {
    let mut s = Session::new();
    run(
        &mut s,
        &[
            ("geometry.solid", json!({"shape": "box", "size": [160, 5, 5]})),
            ("project.analysis", json!({"kind": "particles"})),
            ("particles.walls", json!({"x": "periodic", "y": "periodic", "z": "periodic"})),
            ("particles.plasma", json!({"density": 1e16, "per_cell": [16, 1, 1], "wave": 1e4})),
            ("particles.probe", json!({"at": [40, 0, 0]})),
            ("project.setup", json!({"time": 12, "cell": 5})),
        ],
    );
    s
}

#[test]
fn a_cold_plasma_rings_at_the_plasma_frequency() {
    let mut s = plasma();
    let steps = crate::workflow::steps(&s);
    assert!(steps[1..4].iter().all(|st| st.status == crate::workflow::Status::Done), "{steps:?}");
    let r = s.execute("solve.run", &json!({})).unwrap();
    // ω_p/2π = √(n e²/ε₀ m)/2π = 0.8979 GHz; the grid's shape and the leapfrog put it 0.23 % low.
    let f = r["probes"][0]["frequency"].as_f64().unwrap();
    assert!((f / 0.897_866 - 1.0 + 0.0023).abs() < 3e-4, "{f} GHz");
    assert_eq!((r["cells"].clone(), r["particles"].as_u64()), (json!([32, 1, 1]), Some(512)));
    let kept = s.results.particles.clone().unwrap();
    assert!(kept.history.len() > 100 && kept.snapshot.len() == 512 && kept.probes[0].e.len() == kept.steps);
}

#[test]
fn a_planar_diode_draws_the_space_charge_limited_current() {
    let mut s = Session::new();
    run(
        &mut s,
        &[
            ("geometry.solid", json!({"shape": "box", "size": [0.25, 0.25, 10]})),
            ("project.analysis", json!({"kind": "particles"})),
            ("particles.walls", json!({"x": "periodic", "y": "periodic"})),
            ("particles.voltage", json!({"axis": "z", "volts": 1e5})),
            ("particles.emitter", json!({"face": "z-"})),
        ],
    );
    let r = s.execute("solve.particles", &json!({"time": 1, "cell": 0.25})).unwrap();
    // Jory & Trivelpiece's current at 100 kV across 1 cm, 7.23e5 A/m², 5.9 % high at 40 cells
    // (emcraft_fdtd::pic's convergence test); electrons arriving: a negative current.
    let want = emcraft_theory::plasma::child_langmuir_relativistic(1e5, 1e-2, emcraft_particles::ELECTRON).unwrap();
    let j = -r["walls"]["z+"]["density"].as_f64().unwrap();
    assert!((j / want - 1.059).abs() < 0.01, "{j} A/m²");
    assert!(r["walls"]["z-"]["current"].as_f64() == Some(0.0) && r["walls"].get("x+").is_none(), "{r}");
    let e = r["fastest_ev"].as_f64().unwrap();
    assert!(e < 1e5 && e > 0.9e5, "the fastest electrons near the anode: {e} eV");
}

#[test]
fn sources_are_checked_kept_and_undone() {
    let mut s = Session::new();
    s.execute("geometry.solid", &json!({"shape": "box", "size": [10, 10, 10]})).unwrap();
    for (c, p) in [
        ("particles.walls", json!({"x": "sideways"})),
        ("particles.emitter", json!({"face": "w+"})),
        ("particles.emitter", json!({"face": "z-", "per_node": 0})),
        ("particles.emitter", json!({"face": "z-", "current": -1})),
        ("particles.plasma", json!({"density": -1})),
        ("particles.plasma", json!({"density": 1e16, "per_cell": [0, 1, 1]})),
        ("particles.plasma", json!({"density": 1e16, "per_cell": [64, 64, 64]})),
        ("particles.plasma", json!({"density": 1e16, "per_cell": "four"})),
        ("particles.plasma", json!({"density": 1e16, "drift": [1e300, 0, 0]})),
        ("particles.voltage", json!({"axis": "w", "volts": 1})),
        ("particles.magnetic", json!({"b": [f64::MAX, 0, 0]})),
        ("particles.probe", json!({"at": [0, 0]})),
    ] {
        assert!(s.execute(c, &p).is_err(), "{c} {p}");
    }
    assert!(s.model.particles.is_empty());
    // Nothing to run yet; then an emitter on a wall that is periodic is refused at the run.
    assert!(s.execute("solve.particles", &json!({"time": 1, "cell": 1})).unwrap_err().to_string().contains("nothing puts particles"));
    s.execute("particles.walls", &json!({"z": "periodic"})).unwrap();
    s.execute("particles.emitter", &json!({"face": "z-"})).unwrap();
    let e = s.execute("solve.particles", &json!({"time": 1, "cell": 1})).unwrap_err().to_string();
    assert!(e.contains("z-"), "{e}");
    // A voltage across walls with others closed is refused, said how to fix.
    s.execute("particles.walls", &json!({"z": "conductor"})).unwrap();
    s.execute("particles.voltage", &json!({"axis": "z", "volts": 10})).unwrap();
    let e = s.execute("solve.particles", &json!({"time": 1, "cell": 1})).unwrap_err().to_string();
    assert!(e.contains("periodic"), "{e}");
    // Too many steps, a hostile cell.
    assert!(s.execute("solve.particles", &json!({"time": 1e12, "cell": 1})).is_err());
    assert!(s.execute("solve.particles", &json!({"time": 1, "cell": 1e-9})).is_err());
    assert!(s.execute("solve.particles", &json!({"time": f64::NAN, "cell": 1})).is_err());
    // Cells finer along z only in a body of revolution: refused when confirmed, as when solved
    // (a setup once held it and failed only at its solve).
    let e = s.execute("project.setup", &json!({"kind": "particles", "time": 1, "cell": 1, "fine": 4})).unwrap_err().to_string();
    assert!(e.contains("body of revolution"), "{e}");
    assert!(s.execute("project.setup", &json!({"kind": "particles", "time": 1, "cell": 1, "axisymmetric": true, "fine": 17})).is_err());
    s.execute("edit.undo", &json!({})).unwrap();
    assert!(s.model.particles.voltage.is_none());
    s.execute("particles.clear", &json!({})).unwrap();
    assert!(s.model.particles.emitters.is_empty() && s.model.particles.walls[2] == emcraft_model::particles::Wall::Conductor);
}

#[test]
fn a_run_saved_opens_with_its_sources_and_result() {
    let mut s = plasma();
    s.execute("solve.particles", &json!({"time": 1})).unwrap();
    let path = scratch("plasma.emcraft");
    s.execute("file.save", &json!({"path": path})).unwrap();
    let mut t = Session::new();
    t.execute("file.open", &json!({"path": path})).unwrap();
    assert_eq!(t.model.particles, s.model.particles);
    assert_eq!(t.model.analysis(), emcraft_model::Analysis::Particles);
    assert_eq!(t.results.particles, s.results.particles, "read back, not solved again");
    // A change to the sources drops the run.
    t.execute("particles.magnetic", &json!({"b": [0, 0, 0.1]})).unwrap();
    assert!(t.results.particles.is_none());
    let mut bad = (*s.results.particles.clone().unwrap()).clone();
    bad.dt = f64::NAN;
    assert!(bad.check().is_err());
    // Every number is checked, not only the grid's (a probe's sample was once let through).
    let mut bad = (*s.results.particles.clone().unwrap()).clone();
    bad.probes[0].e[0][0] = f64::INFINITY;
    assert!(bad.check().is_err());
    let mut bad = (*s.results.particles.clone().unwrap()).clone();
    bad.history[0].field = f64::NAN;
    assert!(bad.check().is_err());
}

#[test]
fn a_bunch_through_a_closed_pillbox_leaves_its_loss_factor() {
    // The perfectly conducting pillbox's outside as metal cells, a 1 nC bunch (σ_z = 2 mm, 10 GeV)
    // through it on the axis: q² Σ k_n e^{−(ω_n σ_t)²} left in it (P. B. Wilson, SLAC-PUB-4547,
    // 1989, §3; emcraft_theory::wake).
    let mut s = Session::new();
    run(
        &mut s,
        &[
            ("geometry.solid", json!({"shape": "cylinder", "radius": 10, "length": 13.06})),
            ("project.analysis", json!({"kind": "particles"})),
            ("particles.bunch", json!({"face": "z-", "charge": 1, "length": 2, "energy": 1e10})),
        ],
    );
    let r = s.execute("solve.particles", &json!({"time": 0.12, "cell": 0.5})).unwrap();
    let want = emcraft_theory::wake::Pillbox::new(0.01, 0.01306).loss_factor(1.0, 2e-3 / 299_792_458.0).unwrap() * 1e-12;
    let k = r["loss_factor_v_per_pc"].as_f64().unwrap();
    assert!((k / want - 1.0).abs() < 4e-3, "{k} V/pC against {want}");
    assert_eq!((r["particles"].as_u64(), r["walls"]["z+"]["current"].as_f64().map(f64::is_sign_negative)), (Some(0), Some(true)));
    // Hostile bunches are refused.
    for p in [
        json!({"face": "z-", "charge": -1, "length": 2}),
        json!({"face": "z-", "charge": 1, "length": 0}),
        json!({"face": "q", "charge": 1, "length": 1}),
    ] {
        assert!(s.execute("particles.bunch", &p).is_err(), "{p}");
    }
}

#[test]
fn particle_runs_take_lossless_dielectrics_only() {
    // A lossy filling is refused rather than run as if lossless.
    let mut s = Session::new();
    run(
        &mut s,
        &[
            ("model.material.create", json!({"name": "lossy", "eps_r": 2.25, "loss_tangent": 0.01})),
            ("geometry.solid", json!({"shape": "cylinder", "radius": 10, "length": 13.06, "material": "lossy"})),
            ("particles.bunch", json!({"face": "z-", "charge": 0.001, "length": 2})),
        ],
    );
    let e = s.execute("solve.particles", &json!({"time": 0.1, "cell": 1})).unwrap_err().to_string();
    assert!(e.contains("lossy") && e.contains("not supported yet"), "{e}");
}

#[test]
fn beams_are_checked_and_reply_their_gamma() {
    let mut s = Session::new();
    s.execute("geometry.solid", &json!({"shape": "cylinder", "radius": 5, "length": 10})).unwrap();
    for p in [
        json!({"face": "z-", "current": -1, "energy": 1e5}),
        json!({"face": "z-", "current": 1, "energy": 0}),
        json!({"face": "z-", "current": 1, "energy": 1e5, "radius": 1, "inner": 2}),
        json!({"face": "z-", "current": 1, "energy": 1e5, "rise": -5}),
    ] {
        assert!(s.execute("particles.beam", &p).is_err(), "{p}");
    }
    s.execute("particles.beam", &json!({"face": "z-", "current": 10, "energy": 1e5, "radius": 2})).unwrap();
    let r = s.execute("solve.particles", &json!({"time": 0.1, "cell": 1})).unwrap();
    // 100 keV: γ = 1.196 at the start, a little less where the beam has got to.
    let g = r["mid_gamma"].as_f64().unwrap();
    assert!(g > 1.18 && g < 1.1958, "{g}");
}

#[test]
fn an_open_face_lets_a_beam_and_its_field_out() {
    // A beam down a tube whose far end is open: it leaves through the absorbing layer (counted on
    // the wall behind it); an emitter on an open face, or an open face on a periodic axis, is
    // refused.
    let mut s = Session::new();
    run(
        &mut s,
        &[
            ("geometry.solid", json!({"shape": "cylinder", "radius": 5, "length": 20})),
            ("particles.magnetic", json!({"b": [0, 0, 5]})),
            ("particles.beam", json!({"face": "z-", "current": 5, "energy": 1e5, "radius": 2, "rise": 20})),
            ("particles.open", json!({"face": "z+"})),
        ],
    );
    let r = s.execute("solve.particles", &json!({"time": 0.4, "cell": 1})).unwrap();
    let out = -r["walls"]["z+"]["current"].as_f64().unwrap();
    assert!((out / 5.0 - 1.0).abs() < 0.05, "{out} A out of 5 A: {r}");
    assert_eq!(r["cells"], json!([10, 10, 30]), "the layer's 10 cells beyond z+");
    s.execute("particles.emitter", &json!({"face": "z+"})).unwrap();
    assert!(s.execute("solve.particles", &json!({"time": 0.1, "cell": 1})).unwrap_err().to_string().contains("open"));
    s.execute("edit.undo", &json!({})).unwrap();
    s.execute("particles.walls", &json!({"z": "periodic"})).unwrap();
    assert!(s.execute("solve.particles", &json!({"time": 0.1, "cell": 1})).unwrap_err().to_string().contains("periodic"));
    s.execute("particles.open", &json!({"face": "z+", "closed": true})).unwrap();
    assert!(s.model.particles.open.is_empty());
}

#[test]
fn a_bunch_through_a_pillbox_as_a_body_of_revolution_leaves_its_loss_factor() {
    // The closed pillbox of the Cartesian case run on the r–z grid (Wilson, SLAC-PUB-4547, 1989,
    // §3; emcraft_theory::wake): the same loss factor, the box the cylinder's.
    let mut s = Session::new();
    run(
        &mut s,
        &[
            ("geometry.solid", json!({"shape": "cylinder", "radius": 10, "length": 13.06})),
            ("project.analysis", json!({"kind": "particles"})),
            ("particles.bunch", json!({"face": "z-", "charge": 1, "length": 2, "energy": 1e10})),
            ("particles.probe", json!({"at": [5, 0, 6.5]})),
        ],
    );
    let r = s.execute("solve.particles", &json!({"time": 0.12, "cell": 0.5, "axisymmetric": true})).unwrap();
    let want = emcraft_theory::wake::Pillbox::new(0.01, 0.01306).loss_factor(1.0, 2e-3 / 299_792_458.0).unwrap() * 1e-12;
    let k = r["loss_factor_v_per_pc"].as_f64().unwrap();
    assert!((k / want - 1.0).abs() < 4e-3, "{k} V/pC against {want}");
    assert_eq!(r["cells"], json!([40, 40, 26]));
    assert!(r["walls"]["z+"]["current"].as_f64().is_some_and(f64::is_sign_negative) && r["walls"].get("x-").is_none(), "{r}");
    let kept = s.results.particles.clone().unwrap();
    assert!(kept.axisymmetric && (kept.area(2) / (std::f64::consts::PI * 1e-4) - 1.0).abs() < 1e-12);
}

#[test]
fn an_output_plane_measures_the_power_a_ring_beams_own_field_carries() {
    // A thin ring beam (radius r_b) in a tube of radius a carries power in its own static field:
    // E_r = λ/(2πε₀r) and H_θ = I/(2πr) between the ring and the wall, so Poynting's flux is
    // P = I² ln(a/r_b)/(2πε₀v) (Jackson §§2.8, 5.3: the line charge's and the current's fields).
    // 20 A at 100 keV, 5 mm in 10 mm: about 30 kW, its space charge slowing it by 0.6 %.
    let mut s = Session::new();
    run(
        &mut s,
        &[
            ("geometry.solid", json!({"shape": "cylinder", "radius": 10, "length": 60})),
            ("project.analysis", json!({"kind": "particles"})),
            ("particles.magnetic", json!({"b": [0, 0, 3]})),
            ("particles.beam", json!({"face": "z-", "current": 20, "energy": 1e5, "radius": 5.1, "inner": 4.9, "rise": 100})),
            ("particles.open", json!({"face": "z+"})),
            ("particles.output", json!({"at": 40})),
            ("particles.probe", json!({"at": [7, 0, 30]})),
        ],
    );
    let r = s.execute("solve.particles", &json!({"time": 2, "cell": 0.5, "axisymmetric": true, "frames": 4})).unwrap();
    let (eps0, c0, mc2) = (8.854_187_812_8e-12, 299_792_458.0, 510_998.95);
    let gamma: f64 = 1.0 + 1e5 / mc2;
    let v = c0 * (1.0 - 1.0 / (gamma * gamma)).sqrt();
    let want = 20.0f64.powi(2) * 2f64.ln() / (2.0 * std::f64::consts::PI * eps0 * v);
    let out = &r["outputs"][0];
    let power = out["power_w"].as_f64().unwrap();
    assert!((power / want - 1.0).abs() < 0.03, "{power} W against {want} W");
    // All of it is the beam's steady field: the oscillations carry next to nothing, and the
    // efficiency (theirs against the beam's 2 MW) is about zero.
    let (steady, oscillating) = (out["steady_w"].as_f64().unwrap(), out["oscillating_w"].as_f64().unwrap());
    assert!((steady / power - 1.0).abs() < 0.01 && oscillating.abs() < 0.01 * power, "{out}");
    let efficiency = out["efficiency"].as_f64().unwrap();
    assert!((efficiency - oscillating / (20.0 * 1e5)).abs() < 1e-12, "{out}");
    let kept = s.results.particles.clone().unwrap();
    assert!(kept.history.iter().all(|h| h.power.len() == 1) && kept.outputs.len() == 1);
    // On the Cartesian grid and on a device, output planes are refused, not left out.
    let cartesian = s.clone().execute("solve.particles", &json!({"time": 0.1, "cell": 1})).unwrap_err().to_string();
    assert!(cartesian.contains("output planes"), "{cartesian}");
    let device = s.clone().execute("solve.particles", &json!({"time": 0.1, "cell": 1, "axisymmetric": true, "device": "cubecl-cpu"}));
    assert!(device.unwrap_err().to_string().contains("output planes"));
    assert!(s.clone().execute("particles.output", &json!({"at": "x"})).is_err());
    // Its records as CSV: the history with the plane's power, the particles' phase space.
    let history = scratch("history.csv");
    let r = s.execute("results.particles.export", &json!({"path": history})).unwrap();
    let text = std::fs::read_to_string(&history).unwrap();
    assert_eq!(text.lines().count(), r["rows"].as_u64().unwrap() as usize + 1);
    let phase = scratch("phase.csv");
    let r = s.execute("results.particles.export", &json!({"path": phase, "what": "particles"})).unwrap();
    assert!(r["rows"].as_u64().unwrap() > 100);
    let row: Vec<f64> = std::fs::read_to_string(&phase).unwrap().lines().nth(1).unwrap().split(',').map(|v| v.parse().unwrap()).collect();
    assert!(row.len() == 8 && row[6] > 9e4 && row[6] < 1.01e5, "{row:?}");
    let probes = scratch("probes.csv");
    let r = s.execute("results.particles.export", &json!({"path": probes, "what": "probes"})).unwrap();
    assert!(r["rows"].as_u64().unwrap() > 100, "{r}");
    // Four moments along the run: their particles and E_z on the r–z plane, in time order.
    let kept = s.results.particles.clone().unwrap();
    assert_eq!(kept.moments.len(), 4);
    assert!(kept.moments.windows(2).all(|w| w[0].t < w[1].t) && (kept.moments[3].t - 2e-9).abs() < 0.01e-9);
    assert!(kept.moments.iter().all(|m| !m.particles.is_empty() && m.particles.len() <= crate::particles::FRAME_PARTICLES));
    let map = kept.moments[3].map.as_ref().unwrap();
    assert!(
        map.ez.len() == map.r.len() * map.z.len()
            && map.ez.len() <= crate::particles::MAP_POINTS
            && map.z[0] < 1e-3
            && map.z[map.z.len() - 1] > 0.059
    );
    let frames = scratch("frames.csv");
    let r = s.execute("results.particles.export", &json!({"path": frames, "what": "frames"})).unwrap();
    let rows = kept.moments.iter().map(|m| m.particles.len()).sum::<usize>();
    assert_eq!(r["rows"].as_u64().unwrap() as usize, rows);
    let maps = scratch("maps.csv");
    let r = s.execute("results.particles.export", &json!({"path": maps, "what": "maps"})).unwrap();
    assert_eq!(r["rows"].as_u64().unwrap() as usize, 4 * map.ez.len());
    assert!(s.clone().execute("solve.particles", &json!({"time": 0.1, "cell": 1, "axisymmetric": true, "frames": 61})).is_err());
    // Moments over the run's last half nanosecond, each at its own step (not at the records').
    let mut late = s.clone();
    late.execute("solve.particles", &json!({"time": 2, "cell": 0.5, "axisymmetric": true, "frames": 4, "frames_from": 1.5})).unwrap();
    let at: Vec<f64> = late.results.particles.as_ref().unwrap().moments.iter().map(|m| m.t * 1e9).collect();
    let dt = late.results.particles.as_ref().unwrap().dt * 1e9;
    assert_eq!(at.len(), 4);
    for (t, want) in at.iter().zip([1.625, 1.75, 1.875, 2.0]) {
        assert!((t - want).abs() <= 1.01 * dt, "{at:?}");
    }
    assert!(s.clone().execute("solve.particles", &json!({"time": 0.1, "cell": 1, "axisymmetric": true, "frames": 2, "frames_from": 0.2})).is_err());
    assert!(s.execute("results.particles.export", &json!({"path": phase, "what": "nope"})).is_err());
    assert!(Session::new().execute("results.particles.export", &json!({"path": phase})).is_err());
}

#[test]
fn a_field_that_falls_away_spreads_its_beam_to_the_wall() {
    // A ring beam follows its field lines: where B_z falls the lines spread (the flux πr²B_z
    // they hold is kept, Jackson §12.5's adiabatic invariant), so a 5 mm beam reaches the 10 mm
    // wall once the field is down to a quarter, short of the far end; in a uniform field it
    // reaches the end.
    let reaching_the_end = |end: Option<f64>| {
        let mut s = Session::new();
        let mut field = json!({"b": [0, 0, 1]});
        if let Some(z) = end {
            field["end"] = json!(z);
            field["fall"] = json!(5);
        }
        run(
            &mut s,
            &[
                ("geometry.solid", json!({"shape": "cylinder", "radius": 10, "length": 80})),
                ("project.analysis", json!({"kind": "particles"})),
                ("particles.magnetic", field),
                ("particles.beam", json!({"face": "z-", "current": 20, "energy": 1e5, "radius": 5.1, "inner": 4.9, "rise": 100})),
                ("particles.open", json!({"face": "z+"})),
            ],
        );
        let r = s.execute("solve.particles", &json!({"time": 2, "cell": 0.5, "axisymmetric": true})).unwrap();
        -r["walls"]["z+"]["current"].as_f64().unwrap_or(0.0)
    };
    let (uniform, falling) = (reaching_the_end(None), reaching_the_end(Some(40.0)));
    assert!(uniform > 19.0 && falling < 0.5, "into the open end: {uniform} A in a uniform field, {falling} A past its end");
    let mut s = Session::new();
    assert!(s.execute("particles.magnetic", &json!({"b": [0, 0, 1], "fall": 5})).is_err());
    assert!(s.execute("particles.magnetic", &json!({"b": [0, 0, 1], "end": 40, "fall": -1})).is_err());
    // The Cartesian grid refuses it rather than leave it out.
    run(
        &mut s,
        &[
            ("geometry.solid", json!({"shape": "cylinder", "radius": 10, "length": 80})),
            ("project.analysis", json!({"kind": "particles"})),
            ("particles.beam", json!({"face": "z-", "current": 20, "energy": 1e5, "radius": 5})),
            ("particles.magnetic", json!({"b": [0, 0, 1], "end": 40})),
        ],
    );
    let err = s.execute("solve.particles", &json!({"time": 0.1, "cell": 2})).unwrap_err().to_string();
    assert!(err.contains("falls away"), "{err}");
}

#[test]
fn a_relativistic_ring_beam_in_a_body_of_revolution_sinks_to_its_closed_form() {
    // Bogdankevich & Rukhadze's ring (Sov. Phys. Usp. 14:163, 1971) at γ₀ = 3 and 0.7 of its
    // limiting current, confirmed as the job's settings: on cells 8 times finer along z its
    // depression is the closed form's to 3.5 % (on square cells the grid's numerical Cherenkov
    // instability deepens it by a quarter, emcraft_fdtd::axi).
    use emcraft_theory::plasma::AnnularBeam;
    let beam = AnnularBeam { gamma0: 3.0, ratio: 10.0 / 6.0, species: emcraft_particles::ELECTRON };
    let current = 0.7 * beam.limiting().unwrap();
    let mut s = Session::new();
    run(
        &mut s,
        &[
            ("geometry.solid", json!({"shape": "cylinder", "radius": 10, "length": 50})),
            ("project.analysis", json!({"kind": "particles"})),
            ("particles.magnetic", json!({"b": [0, 0, 10]})),
            ("particles.beam", json!({"face": "z-", "current": current, "energy": 2.0 * 510_998.95, "radius": 6.25, "inner": 5.75, "rise": 200})),
            ("project.setup", json!({"time": 1.2, "cell": 1, "axisymmetric": true, "fine": 8})),
        ],
    );
    assert_eq!(crate::job::setup_reply(&s)["fine"], json!(8));
    let r = s.execute("solve.run", &json!({})).unwrap();
    let want = 3.0 - beam.depressed(current).unwrap();
    let got = 3.0 - r["mid_gamma"].as_f64().unwrap();
    assert!((got / want - 1.0).abs() < 0.035, "depression {got} against {want}");
}

#[test]
fn a_body_of_revolution_refuses_what_it_cannot_run_yet() {
    let base = || {
        let mut s = Session::new();
        run(
            &mut s,
            &[
                ("geometry.solid", json!({"shape": "cylinder", "radius": 10, "length": 20})),
                ("particles.beam", json!({"face": "z-", "current": 1, "energy": 1e4, "radius": 3})),
            ],
        );
        s
    };
    let axi = json!({"time": 0.1, "cell": 1, "axisymmetric": true});
    for (step, word) in [
        (("particles.emitter", json!({"face": "z-"})), "emitting"),
        (("particles.magnetic", json!({"b": [1, 0, 0]})), "across the axis"),
        (("particles.open", json!({"face": "x+"})), "faces other"),
        (("particles.beam", json!({"face": "z-", "current": 1, "energy": 1e4, "radius": 1, "at": [3, 0, 0]})), "off the axis"),
    ] {
        let mut s = base();
        s.execute(step.0, &step.1).unwrap();
        let e = s.execute("solve.particles", &axi).unwrap_err().to_string();
        assert!(e.contains("not supported yet") && e.contains(word), "{e}");
    }
    let mut s = base();
    run(
        &mut s,
        &[
            ("model.material.create", json!({"name": "glass", "eps_r": 4})),
            ("geometry.solid", json!({"shape": "cylinder", "radius": 2, "length": 5, "material": "glass"})),
        ],
    );
    let e = s.execute("solve.particles", &axi).unwrap_err().to_string();
    assert!(e.contains("dielectric") && e.contains("glass"), "{e}");
    // An open end on cells 8 times finer along z: its layer at most 64 cells (it asked 80).
    let mut s = base();
    s.execute("particles.open", &json!({"face": "z+"})).unwrap();
    assert!(s.execute("solve.particles", &json!({"time": 0.02, "cell": 1, "axisymmetric": true, "fine": 8})).is_ok());
    let mut s = base();
    for p in [
        json!({"time": 0.1, "cell": 1, "fine": 4}),
        json!({"time": 0.1, "cell": 1, "axisymmetric": true, "fine": 17}),
        json!({"time": 0.1, "cell": 1, "axisymmetric": true, "fine": 0}),
    ] {
        assert!(s.execute("solve.particles", &p).is_err(), "{p}");
    }
}

#[test]
fn a_body_of_revolution_runs_on_the_gpu_kernels_as_on_the_threads() {
    // A ring beam into a cylinder along a steady field, on the CPU's threads and on the GPU's
    // kernels (CubeCL's CPU runtime, double precision): the same rings at the end, the same
    // history and probe to rounding (the device sums the current in fixed point, emcraft_gpu).
    let solved = |device: &str| {
        let mut s = Session::new();
        run(
            &mut s,
            &[
                ("geometry.solid", json!({"shape": "cylinder", "radius": 5, "length": 10})),
                ("project.analysis", json!({"kind": "particles"})),
                ("particles.magnetic", json!({"b": [0, 0, 0.5]})),
                ("particles.beam", json!({"face": "z-", "current": 10, "energy": 1e5, "radius": 3, "inner": 1, "rise": 0})),
                ("particles.probe", json!({"at": [2, 1, 5]})),
            ],
        );
        let r = s.execute("solve.particles", &json!({"time": 0.2, "cell": 0.5, "axisymmetric": true, "device": device, "precision": "double"}));
        (r.unwrap(), s.results.particles.clone().unwrap())
    };
    let (cpu, a) = solved("cpu");
    let (gpu, b) = solved("cubecl-cpu");
    assert_eq!((cpu["device"].as_str(), gpu["device"].as_str(), gpu["precision"].as_str()), (Some("cpu"), Some("cubecl-cpu"), Some("double")));
    assert!(a.particles > 50 && a.particles == b.particles, "{} against {}", a.particles, b.particles);
    let close = |x: f64, y: f64| (x - y).abs() <= 1e-9 * x.abs().max(y.abs()).max(1e-300);
    for (p, q) in a.history.iter().zip(&b.history) {
        assert!(close(p.field, q.field) && close(p.kinetic, q.kinetic) && p.particles == q.particles, "{p:?} against {q:?}");
    }
    let top = a.probes[0].e.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
    let worst = a.probes[0].e.iter().flatten().zip(b.probes[0].e.iter().flatten()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    assert!(top > 0.0 && worst <= 1e-9 * top, "{worst:e} of {top:e}");
    // A Cartesian run says it ran on the CPU's threads.
    let mut s = plasma();
    let r = s.execute("solve.particles", &json!({"time": 0.05, "device": "cubecl-cpu"})).unwrap();
    assert_eq!(r["device"], json!("cpu"));
}
