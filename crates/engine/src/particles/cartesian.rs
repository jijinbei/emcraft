//! `solve.particles` on the Cartesian grid on a device (`emcraft_gpu::PicDevice`): its clock and
//! bunches on the host, its fields and particles on the device until [`Run::end`] brings them
//! back; E at the probes and the fields on the output planes recorded there each step, the
//! fields and the particles read back at each moment.

use emcraft_fdtd::pic::{self, Plane, PlaneTaps};
use emcraft_gpu::{Gpu, PicDevice};

use super::{Cartesian, FieldMap, Run, Snap, Stepped, fdtd};
use crate::Result;

/// A Cartesian run on a device, and where its output planes' fields lie.
pub(super) struct OnDevice {
    run: Cartesian,
    device: Option<PicDevice>,
    planes: Vec<PlaneTaps>,
}

impl OnDevice {
    /// `run` moved onto `gpu`, E watched at `probes` and the planes across z at `outputs` (m
    /// from the grid's corner).
    pub(super) fn new(mut run: Cartesian, gpu: Gpu, probes: &[[f64; 3]], outputs: &[f64]) -> Result<OnDevice> {
        let planes: Vec<PlaneTaps> = outputs.iter().map(|z| run.pic.plane_taps(*z)).collect();
        let mut device = gpu.pic(&mut run.pic, probes).map_err(fdtd)?;
        let taps: Vec<pic::Tap> = planes.iter().flat_map(|p| p.taps.iter().copied()).collect();
        device.tap(&taps).map_err(fdtd)?;
        Ok(OnDevice { run, device: Some(device), planes })
    }
}

impl Run for OnDevice {
    fn advance(&mut self) -> Result<pic::Moved> {
        Ok(self.span(1, &[], &[], &|_| false)?.into_iter().next().map_or_else(pic::Moved::default, |s| s.0))
    }
    fn energy(&mut self) -> Result<pic::Energy> {
        match self.device.as_mut() {
            Some(d) => d.energy(&mut self.run.pic).map_err(fdtd),
            None => self.run.energy(),
        }
    }
    fn held(&self) -> usize {
        self.device.as_ref().map_or_else(|| self.run.held(), PicDevice::held)
    }
    /// The field as the host last read it back (the device's own is recorded at the probes).
    fn e_at_point(&self, at: [f64; 3]) -> [f64; 3] {
        self.run.e_at_point(at)
    }
    /// The plane as the host last read the fields back (the device's own is tapped each step).
    fn plane(&self, z: f64) -> Plane {
        self.run.plane(z)
    }
    fn now(&self) -> f64 {
        self.run.now()
    }
    fn snapshot(&self) -> Vec<Snap> {
        self.run.snapshot()
    }
    /// The particles as they stand on the device, read back (the run goes on there), and the
    /// fields into the host's run.
    fn moment(&mut self) -> Result<(Vec<Snap>, Option<FieldMap>)> {
        let Some(device) = self.device.as_ref() else {
            return Ok((self.run.snapshot(), None));
        };
        device.fields(&mut self.run.pic).map_err(fdtd)?;
        // Population by population, as the threads hold them.
        let mut particles = device.particles().map_err(fdtd)?;
        particles.sort_by_key(|(k, _)| *k);
        let species: Vec<_> = self.run.pic.populations().iter().map(|p| p.species).collect();
        let snaps = particles.iter().filter_map(|(k, p)| species.get(*k as usize).map(|s| (*k as usize, *s, p))).collect::<Vec<_>>();
        Ok((self.run.snaps(snaps.len(), snaps.into_iter()), None))
    }
    /// The steps on the device, E at the probes and the fields on the output planes it was made
    /// with.
    fn span(&mut self, steps: usize, _: &[[f64; 3]], outputs: &[f64], sampled: &dyn Fn(usize) -> bool) -> Result<Vec<Stepped>> {
        let Some(device) = self.device.as_mut() else {
            return (0..steps).map(|_| Ok((self.run.advance()?, None, outputs.iter().map(|z| self.run.plane(*z)).collect()))).collect();
        };
        let done = device.advance(&mut self.run.pic, steps).map_err(fdtd)?;
        Ok(done
            .into_iter()
            .enumerate()
            .map(|(b, step)| {
                let planes = super::planes_of(&self.planes, &step.taps);
                (step.moved, sampled(b).then_some(step.probes), planes)
            })
            .collect())
    }
    fn end(&mut self) -> Result<()> {
        match self.device.take() {
            Some(d) => d.finish(&mut self.run.pic).map_err(fdtd),
            None => Ok(()),
        }
    }
}
