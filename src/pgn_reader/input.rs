use std::ffi::OsStr;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use anyhow::Context;
use bzip2::read::MultiBzDecoder;

use super::*;

pub enum PgnInput<'a> {
    File(PathBuf),
    Bz2 { name: PathBuf, data: &'a [u8] },
}

impl PgnInput<'_> {
    pub fn path(&self) -> &Path {
        match self {
            PgnInput::File(path) | PgnInput::Bz2 { name: path, .. } => path,
        }
    }

    pub fn file_name(&self) -> Option<&OsStr> {
        self.path().file_name()
    }

    pub fn read(&self) -> anyhow::Result<Vec<GameRecord>> {
        match self {
            PgnInput::File(path) => read_pgn(File::open(path)?),
            // &[u8] implements Read
            PgnInput::Bz2 { data, .. } => read_pgn(MultiBzDecoder::new(*data)),
        }
    }
    
    pub fn output_path(
        &self,
        out_dir: &Path,
        format: output::OutputFormat,
    ) -> anyhow::Result<PathBuf> {
        let name = self
            .file_name()
            .with_context(|| format!("{} has no file name", self.path().display()))?;
        let mut path = out_dir.join(name);
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("bz2"))
        {
            path.set_extension(""); // drop ".bz2"
        }
        Ok(path.with_extension(format.extension()))
    }
}

/// A single file, or every .pgn file directly inside a directory
pub fn collect_pgn_files(input: &Path) -> anyhow::Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    anyhow::ensure!(input.is_dir(), "{} does not exist", input.display());

    let mut files: Vec<PathBuf> = std::fs::read_dir(input)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("pgn"))
        })
        .collect();
    files.sort();
    Ok(files)
}

pub fn read_tar_entries(tar_path: &Path) -> anyhow::Result<Vec<(PathBuf, Vec<u8>)>> {
    let mut archive = tar::Archive::new(File::open(tar_path)?);
    let mut blobs = Vec::new();

    for entry in archive.entries()? {
        let mut entry = entry?;
        let name = entry.path()?.into_owned();
        let is_bz2 = name
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("bz2"));
        if !entry.header().entry_type().is_file() || !is_bz2 {
            continue;
        }
        let mut data = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut data)?;
        blobs.push((name, data));
    }
    Ok(blobs)
}
