use config::{Config, ConfigError, Environment, File};
use serde::Deserialize;
use anyhow::{Result, Error};
use std::fmt;

const CONFIG_FILE_PATH: &str = "src/config.toml";
const CONFIG_FILE_PREFIX: &str = "./config/";

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
    resources:String,
    documents:String,
    checkpoints:String,
}


#[derive(Debug, Deserialize, Clone)]
pub struct Settings {
    pub server: Server,
    pub localpath: LocalPath,
    pub log: Log,
}


impl Settings {
    pub fn new() -> Result<Self, Error> {
        let settings = Config::builder()
            .add_source(config::File::with_name(CONFIG_FILE_PATH))
            .add_source(config::Environment::with_prefix("APP"))
            .build()
            .unwrap();
        Ok(settings.try_deserialize::<Settings>().unwrap())

    }
}
