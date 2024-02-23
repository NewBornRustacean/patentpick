use anyhow::{Error, Result};
use mpnet_rs::mpnet::{get_embeddings, load_model};
use opensearch::http::transport::{SingleNodeConnectionPool, TransportBuilder};
use opensearch::http::Url;
use opensearch::indices::IndicesCreateParts;
use opensearch::OpenSearch;
use serde_json::json;

pub fn connect_to_opensearch(url: &str) -> Result<OpenSearch, Error> {
    let url = Url::parse(url)?;
    let conn_pool = SingleNodeConnectionPool::new(url);
    let transport = TransportBuilder::new(conn_pool).disable_proxy().build()?;
    let client = OpenSearch::new(transport);
    Ok(client)
}

pub async fn create_index(client: &OpenSearch) -> Result<()> {
    let response = client
        .indices()
        .create(IndicesCreateParts::Index("patents"))
        .body(json!({
            "mappings" : {
                "properties" : {
                    "title" : { "type" : "text" },
                    "embedding" :{
                        "type": ""
                    }
                }
            }
        }))
        .send()
        .await?;
    Ok(())
}

pub fn bulk_upload_data_opensearch() {}
pub fn get_topk_hits_from_vectordb(client: &OpenSearch, input_query: String) {}
