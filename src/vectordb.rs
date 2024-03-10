use anyhow::{Error, Result};
use qdrant_client::qdrant::SearchResponse;
use qdrant_client::{
    prelude::{Payload, QdrantClient},
    qdrant::{
        vectors_config::Config, with_payload_selector::SelectorOptions, CreateCollection, Distance, Filter,
        PointStruct, ScoredPoint, SearchPoints, VectorParams, VectorsConfig, WithPayloadSelector, WithVectorsSelector
    },
};
use serde_json::json;
use uuid::Uuid;

use crate::documents::PatentRecord;

pub struct VectorDB {
    pub client: QdrantClient,
}

impl VectorDB {
    pub fn new(client_url: &str) -> Self {
        Self {
            client: QdrantClient::from_url(client_url).build().unwrap(),
        }
    }

    pub async fn create_collection(&self, collection_name: &str, vector_dim: u64) -> Result<()> {
        self.client
            .create_collection(&CreateCollection {
                collection_name: collection_name.to_string(),
                vectors_config: Some(VectorsConfig {
                    config: Some(Config::Params(VectorParams {
                        size: vector_dim,
                        distance: Distance::Cosine.into(),
                        hnsw_config: None,
                        quantization_config: None,
                        on_disk: None,
                    })),
                }),
                ..Default::default()
            })
            .await?;

        Ok(())
    }

    pub async fn upsert_embedding_batch(
        &mut self,
        collection_name: &str,
        patent_records: &[PatentRecord],
        embeddings: &Vec<Vec<f32>>,
        chunk_size: usize,
    ) -> Result<()> {
        let mut points: Vec<PointStruct> = Vec::new();

        for (idx, record) in patent_records.iter().enumerate() {
            let payload: Payload = json!(record)
                .try_into()
                .map_err(|err| format!("json! error {}", err))
                .unwrap();
            points.push(PointStruct::new(Uuid::new_v4().to_string(), embeddings[idx].to_vec(), payload));
        }

        println!("points: {:?}", points);
        self.client
            .upsert_points_batch(collection_name.to_string(), None, points, None, chunk_size)
            .await?;
        Ok(())
    }

    pub async fn search(
        &self,
        collection_name: &str,
        embedding: &[f32],
        limit: u64,
        score_threshold: Option<f32>,
        filter: Option<Filter>,
    ) -> Result<SearchResponse> {

        let search_points = SearchPoints {
            collection_name: collection_name.into(),
            vector: embedding.to_vec(),
            filter: filter,
            limit: limit,
            with_payload: Some(true.into()),
            score_threshold: score_threshold,
            with_vectors: Some(true.into()),
            ..Default::default()
        };

        let search_result = self.client.search_points(&search_points).await?;

        Ok(search_result)
    }
}
