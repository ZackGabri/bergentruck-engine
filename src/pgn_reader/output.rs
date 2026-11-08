use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    str::FromStr,
};

use sfbinpack::CompressedTrainingDataEntryWriter;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    #[default]
    Viriformat,
    Sfbinpack,
}

impl OutputFormat {
    pub fn extension(self) -> &'static str {
        match self {
            Self::Viriformat => "vf",
            Self::Sfbinpack => "binpack",
        }
    }
}

impl FromStr for OutputFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "viriformat" | "viri" | "vf" => Ok(Self::Viriformat),
            "sfbinpack" | "binpack" | "sfbp" | "sf" => Ok(Self::Sfbinpack),
            _ => anyhow::bail!("unrecognised packing format: {s}"),
        }
    }
}

pub enum OutputWriter {
    Viri(BufWriter<File>),
    Sfbinpack(Box<CompressedTrainingDataEntryWriter<BufWriter<File>>>),
}

impl OutputWriter {
    pub fn create(path: &Path, format: OutputFormat) -> anyhow::Result<Self> {
        let buffer = BufWriter::new(File::create(path)?);

        Ok(match format {
            OutputFormat::Viriformat => Self::Viri(buffer),
            OutputFormat::Sfbinpack => {
                Self::Sfbinpack(Box::new(CompressedTrainingDataEntryWriter::new(buffer)?))
            }
        })
    }

    pub fn write_game(&mut self, game: &super::GameRecord) -> anyhow::Result<()> {
        match self {
            Self::Viri(writer) => super::viriformat::write_game(writer, game)?,
            Self::Sfbinpack(writer) => super::sfbinpack::write_game(writer, game)?,
        }

        Ok(())
    }

    pub fn finish(&mut self) -> anyhow::Result<()> {
        match self {
            Self::Viri(writer) => writer.flush()?,
            Self::Sfbinpack(writer) => writer.flush_and_end(),
        }

        Ok(())
    }
}
