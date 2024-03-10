mod documents;
mod emails;
mod settings;
mod vectordb;

use std::path::{Path, PathBuf};

use anyhow::{Error, Result};
use chrono::Utc;
use indicatif::ProgressBar;
use mpnet_rs::mpnet::{get_embeddings_parallel, load_model};
use tokenizers::Tokenizer;
use tokio;

use documents::{download_weekly_fulltext, get_abstracts_from_patents, parse_xml};
use emails::{PatentApplicationContent, Subscriber};
use settings::Settings;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let now_utc = Utc::now();
    let today_utc = now_utc.date_naive();
    let settings = Settings::new("src/config.toml").unwrap();
    let (model, mut tokenizer, pooler) = load_model(settings.localpath.checkpoints).unwrap();
    let chunksize: usize = 20;

    // Progress reporting setup
    let progress_bar = ProgressBar::new(3); // 3 steps
    progress_bar.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .expect("REASON")
            .progress_chars("=> "),
    );

    progress_bar.set_message("Downloading XML...");
    let xmlfile_path = download_weekly_fulltext(
        &settings.server.uspto_url, &settings.server.uspto_year, &settings.localpath.documents, &today_utc,
    )
    .await?;
    progress_bar.inc(1);

    progress_bar.set_message("Parsing XML...");
    let patents = parse_xml(xmlfile_path)?;
    progress_bar.inc(1);

    progress_bar.set_message("Getting embeddings...");
    let abstracts = get_abstracts_from_patents(&patents)?;
    let _embeddings = get_embeddings_parallel(&model, &tokenizer, Some(&pooler), &abstracts, chunksize)?;
    let embeddings = _embeddings.to_vec2::<f32>().unwrap();
    progress_bar.inc(1);

    progress_bar.finish_with_message("All steps completed successfully.");
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
