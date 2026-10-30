use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use sfbinpack::CompressedTrainingDataEntryWriter;

#[derive(Default, Debug)]
pub enum OutputFormat {
    #[default]
    Viriformat,
    Sfbinpack,
}

pub enum OutputWriter {
    Viri(BufWriter<File>),
    Sfbinpack(Box<CompressedTrainingDataEntryWriter<BufWriter<File>>>),
}

impl OutputWriter {
    pub fn create(path: &Path, format: OutputFormat) -> anyhow::Result<Self> {
        let extension = match format {
            OutputFormat::Viriformat => "vf",
            OutputFormat::Sfbinpack => "binpack",
        };

        let file = File::create(path.with_extension(extension))?;
        let buffer = BufWriter::new(file);

        match format {
            OutputFormat::Viriformat => Ok(Self::Viri(buffer)),
            OutputFormat::Sfbinpack => Ok(Self::Sfbinpack(Box::new(
                CompressedTrainingDataEntryWriter::new(buffer)?,
            ))),
        }
    }

    pub fn write_game(&mut self, game: &super::GameRecord) -> anyhow::Result<()> {
        match self {
            Self::Viri(writer) => {
                super::viriformat::write_game(writer, game)?;
            }
            Self::Sfbinpack(writer) => {
                super::sfbinpack::write_game(writer, game)?;
            }
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
