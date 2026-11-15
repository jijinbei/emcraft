//! `results.track`: electrons sent through a solved eigenmode (docs/adr/0012-particles.md, stage
//! A), pushed by `emcraft-particles` through the mode's fields at a given accelerating voltage,
//! with an optional uniform static magnetic field.
//!
//! **The field.** The mode's phasors Ê, Ĥ at amplitude a: E(t) = Re(a Ê e^{jωt}),
//! B(t) = μ₀ Re(a Ĥ e^{jωt}) (the particles cross vacuum). |a| sets the field's own voltage along
//! the line, V₀ = |∫ E·dl|, to the voltage asked for; its phase is chosen so that a particle
//! entering at phase φ = 0 and keeping its speed v₀ gains the most: with
//! I = ∫ Ê·dl e^{jωs/v₀}, a particle of charge q entering at t = 0 gains
//! W = q Re(a e^{jφ} I), so a = (V₀/|∫ Ê·dl|)·sign(q)·Ī/|I| gives W = |q| V₀ T cos φ, T = |I|/V₀
//! the transit-time factor (T. P. Wangler, *RF Linear Accelerators*, 2nd ed., Wiley 2008, §2.2).
//! A particle whose speed changes on the way gains what it gains: that is what tracking adds.
//!
//! **Where.** The line runs from the entry plane through `from` to the exit plane through `to`
//! (the region's axis along z by default, as `results.cavity`'s). Past the planes there is no
//! field; between them a point outside the mesh is a wall the particle is lost on. Each particle
//! starts on the entry plane, `offset` off the line, at its phase.

use std::sync::{Arc, Weak};

use emcraft_model::units::MU0;
use emcraft_model::{Point3, Vec3};
use emcraft_particles::{ELECTRON, End, Line, Pusher, Sample, Settings, Start, V3, gamma, track as push_through, velocity};
use emcraft_solver::Complex64 as C;
use emcraft_solver::driven::Driven;
use emcraft_solver::eigen::{Eigen, Mode};
use serde_json::{Value, json};

use crate::accelerating::{Ends, MIN_FIELD_R_OVER_Q, OwnField, images, mode_at, whole_cavity};
use crate::particles::EV;
use crate::{EngineError, Params, Result, Session};

/// The most particles one run sends.
const MAX_PARTICLES: usize = 1000;
/// The most steps one run takes, all its particles together.
const MAX_TOTAL_STEPS: usize = 50_000_000;
/// The highest kinetic energy a particle may start with (eV).
const MAX_ENERGY: f64 = 1e13;
/// The highest voltage a mode is driven to (V).
const MAX_VOLTAGE: f64 = 1e12;
/// The strongest static magnetic field (T).
const MAX_MAGNETIC: f64 = 1e3;
/// The voltage (V) and the starting energy (eV) when not given: a megavolt cavity, MeV electrons.
const DEFAULT_VOLTAGE: f64 = 1e6;
const DEFAULT_ENERGY: f64 = 1e6;
/// Intervals of the line integrals that set the field's amplitude and phase.
const INTERVALS: usize = 400;
/// A particle's step budget: this many transits at its starting speed, and this many periods
/// more (a slowed or turned particle needs longer).
const TRANSITS: f64 = 4.0;
const SPARE_PERIODS: f64 = 2.0;
/// The field is taken to end this fraction of the line short of its planes (a point on a plane
/// is on the mesh's boundary).
const PLANE_MARGIN: f64 = 1e-9;
/// Without a mode asked for, the first whose R/Q is at least this fraction of the best's (a
/// coarse mesh leaves a TE mode some round-off on the axis, R/Q a few thousandths of an ohm,
/// where TM010's is tens).
const OF_THE_BEST: f64 = 0.01;

/// What `results.track` is asked: the mode (none: the first that accelerates), the voltage it
/// is driven to (V), the electrons' energy (eV), entry phases (°) and offsets from the line (m),
/// a static magnetic field (T), the push, steps per period and the most points kept.
struct Ask {
    mode: Option<usize>,
    voltage: f64,
    energy: f64,
    phases: Vec<f64>,
    offsets: Vec<f64>,
    magnetic: V3,
    pusher: Pusher,
    per_period: usize,
    most_points: usize,
}

impl Ask {
    fn read(p: &Params, modes: usize) -> Result<Ask> {
        let mode = p.integer("mode", 1..=modes.max(1)).optional()?;
        let voltage = p.number("voltage").or(DEFAULT_VOLTAGE)?;
        if !(voltage.is_finite() && voltage > 0.0 && voltage <= MAX_VOLTAGE) {
            return Err(p.error("voltage", format!("must be above 0 and at most {MAX_VOLTAGE:e} V")));
        }
        let energy = p.number("energy").or(DEFAULT_ENERGY)?;
        if !(energy.is_finite() && energy > 0.0 && energy <= MAX_ENERGY) {
            return Err(p.error("energy", format!("must be above 0 and at most {MAX_ENERGY:e} eV")));
        }
        let phases: Vec<f64> = match p.number("phase").optional()? {
            Some(deg) if deg.is_finite() => vec![deg],
            Some(_) => return Err(p.error("phase", "must be a finite number of degrees")),
            None => {
                let n = p.integer("phases", 1..=360).or(12)?;
                (0..n).map(|k| 360.0 * k as f64 / n as f64).collect()
            }
        };
        let offsets: Vec<f64> = match p.lengths("offsets").optional()? {
            Some(list) if !list.is_empty() && list.iter().all(|o| o.is_finite()) => list,
            Some(_) => return Err(p.error("offsets", "must be a non-empty list of finite lengths")),
            None => vec![0.0],
        };
        if phases.len() * offsets.len() > MAX_PARTICLES {
            return Err(p.error("phases", format!("phases × offsets: at most {MAX_PARTICLES} particles")));
        }
        let magnetic = match p.vector("magnetic").optional()? {
            Some(b) if b.x.is_finite() && b.y.is_finite() && b.z.is_finite() && b.norm() <= MAX_MAGNETIC => b.to_array(),
            Some(_) => return Err(p.error("magnetic", format!("must be [Bx, By, Bz] in teslas, at most {MAX_MAGNETIC} T"))),
            None => [0.0; 3],
        };
        Ok(Ask {
            mode,
            voltage,
            energy,
            phases,
            offsets,
            magnetic,
            pusher: p.pick::<Pusher>("pusher").or(Pusher::Boris)?,
            per_period: p.integer("steps", 20..=100_000).or(400)?,
            most_points: p.integer("points", 2..=10_000).or(200)?,
        })
    }
}

/// The mode to track through: the one asked for if it accelerates along the line, else the first
/// that does.
fn choose_mode(p: &Params, e: &Eigen, line: Ends, images: f64, asked: Option<usize>) -> Result<usize> {
    let own = |m: &Mode| OwnField::of(e.driven_of(m), m, line, INTERVALS, images);
    if let Some(i) = asked {
        if !own(mode_at(p, e, i)?)?.accelerates() {
            return Err(p.error("mode", format!("mode {i} has no field along the line to accelerate with (a TE mode on the axis?)")));
        }
        return Ok(i);
    }
    let each = e.modes.iter().map(|m| own(m).map(|o| o.r_over_q)).collect::<Result<Vec<f64>>>()?;
    let best = each.iter().copied().fold(0.0, f64::max);
    let floor = MIN_FIELD_R_OVER_Q.max(OF_THE_BEST * best);
    each.iter()
        .position(|rq| *rq >= floor)
        .map(|k| k + 1)
        .ok_or_else(|| EngineError::failed("no mode solved has a field along the line to accelerate with"))
}

/// How a mode is driven: the complex amplitude (of its solved field) whose own voltage along
/// the line is the voltage asked for, phased so that a particle entering at φ = 0 at constant
/// `speed` (m/s) gains the most; and the transit-time factor T at that speed.
struct Drive {
    amplitude: C,
    transit: f64,
}

fn drive(driven: &Driven, m: &Mode, line: Ends, speed: f64, voltage: f64) -> Result<Drive> {
    let omega = 2.0 * std::f64::consts::PI * m.frequency;
    let field_voltage = driven.line_integral(&m.vector, m.frequency, line, INTERVALS, |_| C::new(1.0, 0.0))?;
    let felt = driven.line_integral(&m.vector, m.frequency, line, INTERVALS, |z| C::from_polar(1.0, omega * z / speed))?;
    let v_own = field_voltage.norm();
    if !(v_own > 0.0 && felt.norm() > 0.0) {
        return Err(EngineError::failed("the mode has no field along the line to accelerate with"));
    }
    Ok(Drive { amplitude: C::new(voltage / v_own, 0.0) * ELECTRON.charge.signum() * felt.conj() / felt.norm(), transit: felt.norm() / v_own })
}

/// A unit vector across `axis`: x when the axis is along z, else the axis turned into the
/// plane it makes with z.
fn across(axis: Vec3) -> Vec3 {
    let side = axis.cross(Vec3::new(0.0, 0.0, 1.0));
    if side.norm() < 1e-9 { Vec3::new(1.0, 0.0, 0.0) } else { side.normalized().unwrap_or(Vec3::new(1.0, 0.0, 0.0)) }
}

/// A recorded point of a trajectory: where (m), when (s), its kinetic energy (eV).
#[derive(Clone, Debug, PartialEq)]
pub struct TrackPoint {
    pub x: V3,
    pub t: f64,
    pub energy: f64,
}

/// An electron's run: its entry phase (°) and offset from the line (m), how it ended (and
/// where, lost on a wall), its energy at the end and gained (eV) against the constant-speed
/// gain qV₀T cos φ, its γ, steps and recorded points.
#[derive(Clone, Debug, PartialEq)]
pub struct Tracked {
    pub phase: f64,
    pub offset: f64,
    pub end: End,
    pub energy: f64,
    pub gain: f64,
    pub expected: f64,
    pub gamma: f64,
    pub steps: usize,
    pub points: Vec<TrackPoint>,
}

impl Tracked {
    /// Whether it came out past the exit plane.
    pub fn went_through(&self) -> bool {
        self.end == End::Through
    }
}

/// `results.track`'s result: mode `mode` (from 1) at `frequency` (Hz), the line, the voltage
/// (V) it was driven to and its transit-time factor, the electrons' starting energy (eV), the
/// time step (s), the push, and each electron's run; made on the eigenmodes `of`.
#[derive(Clone, Debug)]
pub struct Tracking {
    pub mode: usize,
    pub frequency: f64,
    pub from: V3,
    pub to: V3,
    pub voltage: f64,
    pub transit: f64,
    pub energy: f64,
    pub time_step: f64,
    pub pusher: Pusher,
    pub particles: Vec<Tracked>,
    of: Weak<Eigen>,
}

impl PartialEq for Tracking {
    /// The same run on the same eigenmodes.
    fn eq(&self, o: &Tracking) -> bool {
        let same = (self.mode, self.frequency, self.from, self.to, self.voltage, self.transit, self.energy, self.time_step, self.pusher)
            == (o.mode, o.frequency, o.from, o.to, o.voltage, o.transit, o.energy, o.time_step, o.pusher);
        same && self.particles == o.particles && Weak::ptr_eq(&self.of, &o.of)
    }
}

impl Tracking {
    /// Whether it was made on the session's eigenmodes as they are now solved.
    pub fn is_current(&self, s: &Session) -> bool {
        s.results().eigen.as_ref().is_some_and(|e| self.of.upgrade().is_some_and(|of| Arc::ptr_eq(&of, e)))
    }

    /// The reply, in display units.
    fn reply(&self, s: &Session) -> Value {
        let particles: Vec<Value> = self
            .particles
            .iter()
            .map(|p| {
                let points: Vec<Value> = p
                    .points
                    .iter()
                    .map(|q| json!([s.show_length(q.x[0]), s.show_length(q.x[1]), s.show_length(q.x[2]), q.t * 1e9, q.energy]))
                    .collect();
                json!({
                    "phase": p.phase,
                    "offset": s.show_length(p.offset),
                    "end": end_name(p.end),
                    "lost_at": match p.end { End::Lost(at) => Some(s.show_point(at)), _ => None },
                    "energy": p.energy,
                    "gain": p.gain,
                    "expected": p.expected,
                    "gamma": p.gamma,
                    "steps": p.steps,
                    "points": points,
                })
            })
            .collect();
        json!({
            "mode": self.mode,
            "frequency": s.show_frequency(self.frequency),
            "from": s.show_point(self.from),
            "to": s.show_point(self.to),
            "voltage": self.voltage,
            "transit": self.transit,
            "energy": self.energy,
            "time_step": self.time_step * 1e9,
            "pusher": self.pusher.id(),
            "particles": particles,
        })
    }
}

/// The end of a run as the reply says it.
fn end_name(end: End) -> &'static str {
    match end {
        End::Through => "through",
        End::Back => "back",
        End::Lost(_) => "lost",
        End::Steps => "steps",
        End::NotFinite => "not_finite",
    }
}

pub fn track(s: &mut Session, p: &Params) -> Result<Value> {
    let e = s.results.eigen.clone().ok_or(EngineError::NotSolved { what: "eigenmodes", run: "solve.eigenmodes" })?;
    whole_cavity(&s.model)?;
    let (from, to) = crate::results::cavity_line(s, p)?;
    let ask = Ask::read(p, e.modes.len())?;
    let index = choose_mode(p, &e, (from, to), images(&s.model), ask.mode)?;
    let m = mode_at(p, &e, index)?;
    let span: Vec3 = to - from;
    let length = span.norm();
    let axis = span.normalized().ok_or_else(|| p.error("to", "must differ from `from`"))?;
    let driven = e.driven_of(m);
    let (x, f) = (&m.vector, m.frequency);
    let omega = 2.0 * std::f64::consts::PI * f;
    let u0 = ELECTRON.momentum(ask.energy * EV, axis.to_array());
    let v0 = velocity(u0);
    let speed = (v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]).sqrt();
    let Drive { amplitude, transit } = drive(driven, m, (from, to), speed, ask.voltage)?;

    let dt = 1.0 / (f * ask.per_period as f64);
    let transit_steps = (length / speed / dt).ceil();
    let max_steps = (TRANSITS * transit_steps + SPARE_PERIODS * ask.per_period as f64).min(emcraft_particles::MAX_STEPS as f64) as usize;
    let count = ask.phases.len() * ask.offsets.len();
    if max_steps.saturating_mul(count) > MAX_TOTAL_STEPS {
        return Err(EngineError::failed(format!(
            "too many steps: {count} particles of up to {max_steps} steps each (at most {MAX_TOTAL_STEPS} in all): fewer phases or offsets, or fewer `steps` per period"
        )));
    }
    let every = ((transit_steps as usize) / ask.most_points).max(1);
    let line = Line { origin: from.to_array(), axis: axis.to_array(), length };
    let side = across(axis);
    let settings = Settings { pusher: ask.pusher, dt, max_steps, every };
    let margin = PLANE_MARGIN * length;
    let static_b = ask.magnetic;
    let mut particles = Vec::with_capacity(count);
    for &offset in &ask.offsets {
        for &phase in &ask.phases {
            let a = amplitude * C::from_polar(1.0, phase.to_radians());
            let fields = |pt: V3, t: f64| {
                let along = line.along(pt);
                if along <= margin || along >= length - margin {
                    return if static_b == [0.0; 3] { Sample::Free } else { Sample::Field([0.0; 3], static_b) };
                }
                let Some((ec, hc)) = driven.fields(x, f, Point3::new(pt[0], pt[1], pt[2])) else { return Sample::Wall };
                let w = a * C::from_polar(1.0, omega * t);
                let re = |c: C| (w * c).re;
                let e = [re(ec.x), re(ec.y), re(ec.z)];
                let b = [MU0 * re(hc.x) + static_b[0], MU0 * re(hc.y) + static_b[1], MU0 * re(hc.z) + static_b[2]];
                Sample::Field(e, b)
            };
            let start = Start { x: (from + side * offset).to_array(), u: u0, t: 0.0 };
            let run = push_through(ELECTRON, start, &fields, line, settings).map_err(|e| EngineError::failed(e.to_string()))?;
            let out = ELECTRON.kinetic(run.u) / EV;
            particles.push(Tracked {
                phase,
                offset,
                end: run.end,
                energy: out,
                gain: out - ask.energy,
                expected: ask.voltage * transit * phase.to_radians().cos(),
                gamma: gamma(run.u),
                steps: run.steps,
                points: run.points.iter().map(|q| TrackPoint { x: q.x, t: q.t, energy: ELECTRON.kinetic(q.u) / EV }).collect(),
            });
        }
    }
    let tracking = Tracking {
        mode: index,
        frequency: f,
        from: from.to_array(),
        to: to.to_array(),
        voltage: ask.voltage,
        transit,
        energy: ask.energy,
        time_step: dt,
        pusher: ask.pusher,
        particles,
        of: Arc::downgrade(&e),
    };
    let reply = tracking.reply(s);
    s.views.track = Some(p.value().clone());
    s.views.tracked = Some(Arc::new(tracking));
    Ok(reply)
}
