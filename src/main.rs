mod documents;
mod emails;
mod opensearch_handler;
mod settings;

use std::path::{Path, PathBuf};

use anyhow::{Error, Result};
use chrono::Utc;
use mpnet_rs::mpnet::{get_embeddings_parallel, load_model};
use tokenizers::Tokenizer;
use tokio;

use documents::{download_weekly_fulltext, parse_xml, get_abstracts_from_patents};
use emails::{PatentApplicationContent, Subscriber};
use settings::Settings;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let now_utc = Utc::now();
    let today_utc = now_utc.date_naive();
    let settings = Settings::new("src/config.toml").unwrap();
    let (model, mut tokenizer, pooler) = load_model(settings.localpath.checkpoints).unwrap();
    let chunksize:usize = 10;

    // 1. download the latest xml into resource/documents/
    let xmlfile_path = download_weekly_fulltext(
        &settings.server.uspto_url,
        &settings.server.uspto_year,
        &settings.localpath.documents,
        &today_utc,
    )
    .await?;

    // 2. parse xml into vec.
    let patents = parse_xml(xmlfile_path);

    // 3. get emgeddings from patents
    let abstracts = get_abstracts_from_patents(&patents.unwrap())?;
    // get_embeddings_parallel(model, tokenizer, pooler, chunksize)

    Ok(())
    // let mut subscriber_seom =Subscriber::new(
    //     "SeomKim".to_string(),
    //     "huiseomkim@gmail.com".to_string(),
    //     vec!["new chemical that targets glucagon like peptide-1".to_string()],
    //     None
    // );
    //
    // let mut mock_results = Vec::new();
    // mock_results.push(emails::PatentApplicationContent::new(
    //     "Rapid transformation of monocot leaf explants".to_string(),
    //     "https://patents.google.com/patent/US20240002870A1/en?oq=US+20240002870+A1".to_string())
    // );
    //
    // mock_results.push(emails::PatentApplicationContent::new(
    //     "PYRIDO[2,3-D]PYRIMIDIN-4-AMINES AS SOS1 INHIBITORS".to_string(),
    //     "https://patents.google.com/patent/US20230357239A1/en?oq=US+20230357239+A1".to_string())
    // );
    //
    // mock_results.push(emails::PatentApplicationContent::new(
    //     "PRIME EDITING GUIDE RNAS, COMPOSITIONS THEREOF, AND METHODS OF USING THE SAME".to_string(),
    //     "https://patents.google.com/patent/US20230357766A1/en?oq=US+20230357766+A1".to_string())
    // );
    //
    // mock_results.push(emails::PatentApplicationContent::new(
    //     "IMAGE SENSOR".to_string(),
    //     "https://patents.google.com/patent/US20230352510A1/en?oq=US+20230352510+A1".to_string())
    // );
    //
    // mock_results.push(emails::PatentApplicationContent::new(
    //     "Expressing Multicast Groups Using Weave Traits".to_string(),
    //     "https://patents.google.com/patent/US20230336371A1/en?oq=US+20230336371+A1".to_string())
    // );
    //
    // subscriber_seom.compose_html(&mock_results).send_email().unwrap();
}
