use anyhow::{anyhow, Error, Result};
use config::{Config, ConfigError, Environment, File};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct Log {
    pub level: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Server {
    pub uspto_url: String,
    pub uspto_pdf_url: String,
    pub uspto_year: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LocalPath {
    pub resources: String,
    pub documents: String,
    pub checkpoints: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct VectorDataBase {
    pub qdrant_url: String,
    pub vector_dim: u64,
    pub collection_name: String,
    pub upload_chunk_size: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Settings {
    pub server: Server,
    pub localpath: LocalPath,
    pub vectordb: VectorDataBase,
}

impl Settings {
    pub fn new(config_file_path: &str) -> Result<Self, Error> {
        if !Path::new(&config_file_path).exists() {
            return Err(anyhow!("Config file not found: {config_file_path}"));
        }

        let settings = Config::builder()
            .add_source(config::File::with_name(config_file_path))
            .add_source(config::Environment::with_prefix("APP"))
            .build()
            .unwrap();
        Ok(settings.try_deserialize::<Settings>().unwrap())
    }
}
