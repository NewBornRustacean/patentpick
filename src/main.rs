mod documents;
mod emails;
mod settings;
mod vectordb;

use std::fs;
use std::path::Path;

use anyhow::{Error, Result};
use chrono::Utc;
use indicatif::ProgressBar;
use mpnet_rs::mpnet::{get_embeddings, get_embeddings_parallel, load_model, normalize_l2};
use qdrant_client::qdrant::{Condition, Filter};
use tokio;

use crate::emails::get_patent_application_contents;
use documents::{download_weekly_fulltext, get_abstracts_from_patents, parse_xml};
use emails::{PatentApplicationContent, Subscriber};
use settings::Settings;
use vectordb::VectorDB;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let now_utc = Utc::now();
    let today_utc = now_utc.date_naive();
    let settings = Settings::new("src/config.toml").unwrap();
    let (model, mut tokenizer, pooler) = load_model(settings.localpath.checkpoints).unwrap();
    let parallel_embedding_chunksize: usize = 20;
    let search_limit: u64 = 10;
    let search_threshold = 0.8f32;

    // Progress reporting setup
    let progress_bar = ProgressBar::new(3); // 3 steps
    progress_bar.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .expect("REASON")
            .progress_chars("=> "),
    );

    progress_bar.set_message("1. Downloading XML...");
    let xmlfile_path = download_weekly_fulltext(
        &settings.server.uspto_url, &settings.server.uspto_year, &settings.localpath.documents, &today_utc,
    )
    .await?;
    progress_bar.inc(1);

    progress_bar.set_message("2. Parsing XML...");
    let patents = parse_xml(xmlfile_path)?;
    progress_bar.inc(1);

    progress_bar.set_message("3. Getting embeddings...");
    let abstracts = get_abstracts_from_patents(&patents)?;
    let _embeddings =
        get_embeddings_parallel(&model, &tokenizer, Some(&pooler), &abstracts, parallel_embedding_chunksize)?;
    let l2norm_embeds = normalize_l2(&_embeddings).unwrap();
    let embeddings = l2norm_embeds.to_vec2::<f32>().unwrap();
    progress_bar.inc(1);

    progress_bar.set_message("4. Uploading embeddings to vectorDB...");
    let collection_name = settings.vectordb.collection_name.as_str();
    let mut vectordb = VectorDB::new(settings.vectordb.qdrant_url.as_str());
    if !vectordb.client.has_collection(collection_name.to_string()).await? {
        vectordb
            .create_collection(collection_name, settings.vectordb.vector_dim)
            .await?;
    }
    vectordb
        .upsert_embedding_batch(collection_name, &patents, &embeddings, settings.vectordb.upload_chunk_size)
        .await?;
    progress_bar.inc(1);

    progress_bar.set_message("5. Sending emails...");
    let json_str =
        fs::read_to_string(Path::new(settings.localpath.resources.as_str()).join("subscribers/subscribers.json"))?;
    let mut subscribers: Vec<Subscriber> = serde_json::from_str(json_str.as_str())?;


    for subscriber in subscribers.iter_mut() {
        let _query_embedding = get_embeddings(
            &model,
            &tokenizer,
            Some(&pooler),
            &subscriber.search_queries.clone().iter().map(|s| s.as_str()).collect(),
        )?;
        let normalizes_query = normalize_l2(&_query_embedding).unwrap();

        // assume that there is only one search query for each subscriber
        let query_to_vec = normalizes_query.to_vec2::<f32>()?;

        let publication_date = patents.get(0).unwrap().publication_date.clone();
        let filter = Some(Filter::must([
            Condition::matches("publication_date", publication_date),
        ]));

        let result = vectordb
            .search(collection_name, query_to_vec.get(0).unwrap(), search_limit, Some(search_threshold), filter)
            .await?;

        let patent_application_contents =
            get_patent_application_contents(&result.result, &settings.server.uspto_pdf_url)?;

        subscriber.compose_html(&patent_application_contents).send_email().unwrap();
    }

    progress_bar.finish_with_message("All steps completed successfully.");
    Ok(())
}
