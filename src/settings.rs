use std::path::Path;
use config::{Config, ConfigError, Environment, File};
use serde::Deserialize;
use anyhow::{Result, Error, anyhow};

#[derive(Debug, Deserialize, Clone)]
pub struct Log {
    pub level: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Server {
    pub opensearch_url: String,
    pub uspto_url:String,
    pub uspto_year:String
}

#[derive(Debug, Deserialize, Clone)]
pub struct LocalPath {
    pub resources:String,
    pub documents:String,
    pub checkpoints:String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Settings {
    pub server: Server,
    pub localpath: LocalPath,
    pub log: Log,
}


impl Settings {
    pub fn new(config_file_path:&str) -> Result<Self, Error> {
        if !Path::new(&config_file_path).exists(){
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

