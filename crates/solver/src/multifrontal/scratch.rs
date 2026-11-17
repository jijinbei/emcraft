//! The scratch file a factorization writes what it keeps out of memory to (out-of-core).

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use num_complex::Complex64 as C;

use super::FactorError;

/// The bytes of an entry in the scratch file: its real part, then its imaginary part, each a
/// little-endian f64.
const ON_DISK: usize = 16;

/// A scratch file on disk for what a factorization keeps out of memory (out-of-core): the
/// factor's columns beyond their budget, or the update matrices waiting for their parent. The
/// file is unlinked as soon as it is made, so it goes when its owner does.
#[derive(Debug)]
pub(crate) struct Scratch {
    file: Mutex<std::fs::File>,
    /// Bytes written.
    pub end: u64,
}

/// The directories for scratch files, in the order tried: `EMCRAFT_SCRATCH`, then
/// `~/.cache/emcraft` (on disk; the system's temporary directory is often in memory), then the
/// temporary directory. A directory that cannot be used (a drive not mounted, no permission)
/// passes to the next.
fn scratch_dirs() -> Vec<std::path::PathBuf> {
    let mut dirs: Vec<std::path::PathBuf> = std::env::var_os("EMCRAFT_SCRATCH").into_iter().map(Into::into).collect();
    if let Some(h) = std::env::var_os("HOME") {
        dirs.push(std::path::Path::new(&h).join(".cache").join("emcraft"));
    }
    dirs.push(std::env::temp_dir());
    dirs
}

fn poisoned<T>(_: T) -> FactorError {
    FactorError::Scratch("the file's lock was poisoned".into())
}

impl Scratch {
    pub(crate) fn new() -> Result<Scratch, FactorError> {
        Scratch::in_first(&scratch_dirs())
    }

    /// A scratch file in the first of `dirs` that takes one; why each refused when none does.
    fn in_first(dirs: &[std::path::PathBuf]) -> Result<Scratch, FactorError> {
        static COUNT: AtomicU64 = AtomicU64::new(0);
        let mut refused = Vec::new();
        for dir in dirs {
            let n = COUNT.fetch_add(1, Ordering::Relaxed);
            let path = dir.join(format!("factor-{}-{n}.bin", std::process::id()));
            let opened = std::fs::create_dir_all(dir).and_then(|_| std::fs::OpenOptions::new().read(true).write(true).create_new(true).open(&path));
            match opened {
                Ok(file) => {
                    // Unlinked at once where the system allows it (the open handle keeps the data).
                    let _ = std::fs::remove_file(&path);
                    return Ok(Scratch { file: Mutex::new(file), end: 0 });
                }
                Err(e) => refused.push(format!("{}: {e}", dir.display())),
            }
        }
        Err(FactorError::Scratch(format!("no directory takes the factor's scratch file ({})", refused.join("; "))))
    }

    /// Appends `v` (`what` it is, for a refused allocation); returns where it went: (byte
    /// offset, entries).
    pub(crate) fn write(&mut self, v: &[C], what: &'static str) -> Result<(u64, usize), FactorError> {
        use std::io::{Seek, SeekFrom, Write};
        let mut bytes = crate::memory::with_capacity(v.len().saturating_mul(ON_DISK), what)?;
        for z in v {
            bytes.extend_from_slice(&z.re.to_le_bytes());
            bytes.extend_from_slice(&z.im.to_le_bytes());
        }
        let mut f = self.file.lock().map_err(poisoned)?;
        f.seek(SeekFrom::Start(self.end)).and_then(|_| f.write_all(&bytes)).map_err(|e| FactorError::Scratch(e.to_string()))?;
        let offset = self.end;
        self.end += bytes.len() as u64;
        Ok((offset, v.len()))
    }

    /// The `len` entries written at byte `offset`.
    pub(crate) fn read(&self, offset: u64, len: usize, what: &'static str) -> Result<Vec<C>, FactorError> {
        use std::io::{Read, Seek, SeekFrom};
        let mut bytes: Vec<u8> = crate::memory::zeros(len.saturating_mul(ON_DISK), what)?;
        let mut f = self.file.lock().map_err(poisoned)?;
        f.seek(SeekFrom::Start(offset)).and_then(|_| f.read_exact(&mut bytes)).map_err(|e| FactorError::Scratch(e.to_string()))?;
        let mut out = crate::memory::with_capacity(len, what)?;
        out.extend(bytes.as_chunks::<ON_DISK>().0.iter().map(|c| {
            let mut re = [0u8; 8];
            let mut im = [0u8; 8];
            re.copy_from_slice(&c[..8]);
            im.copy_from_slice(&c[8..]);
            C::new(f64::from_le_bytes(re), f64::from_le_bytes(im))
        }));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch directory that cannot be made (a drive not mounted) passes to the next; with
    /// none usable the error names each.
    #[test]
    fn an_unusable_scratch_directory_passes_to_the_next() {
        let base = std::env::temp_dir().join(format!("emcraft-scratch-test-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        // A plain file where a directory should be: no directory can be made under it.
        let blocker = base.join("not-a-directory");
        std::fs::write(&blocker, b"x").unwrap();
        let bad = blocker.join("scratch");
        let mut s = Scratch::in_first(&[bad.clone(), base.join("good")]).unwrap();
        let v = [C::new(1.5, -2.0), C::new(0.0, 3.25)];
        let (offset, len) = s.write(&v, "a test").unwrap();
        assert_eq!(s.read(offset, len, "a test").unwrap(), v);
        let e = Scratch::in_first(std::slice::from_ref(&bad)).err().unwrap().to_string();
        assert!(e.contains(&bad.display().to_string()), "{e}");
        let _ = std::fs::remove_dir_all(&base);
    }
}
