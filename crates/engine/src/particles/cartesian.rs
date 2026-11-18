//! `solve.particles` on the Cartesian grid on a device (`emcraft_gpu::PicDevice`): its clock and
//! bunches on the host, its fields and particles on the device until [`Run::end`] brings them
//! back; E at the probes recorded there each step, the fields and the particles read back at
//! each moment.

use emcraft_fdtd::pic;
use emcraft_gpu::{Gpu, PicDevice};

use super::{Cartesian, FieldMap, Run, Snap, Stepped, fdtd};
use crate::Result;

/// A Cartesian run on a device.
pub(super) struct OnDevice {
    run: Cartesian,
    device: Option<PicDevice>,
}

impl OnDevice {
    /// `run` moved onto `gpu`, E watched at `probes` (m from the grid's corner).
    pub(super) fn new(mut run: Cartesian, gpu: Gpu, probes: &[[f64; 3]]) -> Result<OnDevice> {
        let device = gpu.pic(&mut run.pic, probes).map_err(fdtd)?;
        Ok(OnDevice { run, device: Some(device) })
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
    /// The steps on the device and E at the probes it was made with; no output planes (the
    /// Cartesian grid has none).
    fn span(&mut self, steps: usize, _: &[[f64; 3]], _: &[f64], sampled: &dyn Fn(usize) -> bool) -> Result<Vec<Stepped>> {
        let Some(device) = self.device.as_mut() else {
            return (0..steps).map(|_| Ok((self.run.advance()?, None, Vec::new()))).collect();
        };
        let done = device.advance(&mut self.run.pic, steps).map_err(fdtd)?;
        Ok(done.into_iter().enumerate().map(|(b, step)| (step.moved, sampled(b).then_some(step.probes), Vec::new())).collect())
    }
    fn end(&mut self) -> Result<()> {
        match self.device.take() {
            Some(d) => d.finish(&mut self.run.pic).map_err(fdtd),
            None => Ok(()),
        }
    }
}
