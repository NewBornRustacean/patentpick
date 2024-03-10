use anyhow::{Error, Result};
use patentpick::documents::PatentRecord;
use patentpick::vectordb::VectorDB;
use qdrant_client::qdrant::{Condition, CountPoints, Filter};

#[tokio::test]
#[cfg(not(feature = "exclude_from_ci"))]
async fn test_vectordb_insert_search() -> Result<()> {
    let mut vectordb: VectorDB = VectorDB::new("http://localhost:6334");
    let vector_dim = 5;
    let collection_name = "test";

    let patent_records: Vec<PatentRecord> = vec![
        PatentRecord::new(
            "title 111".to_string(),
            "this is abstract 111".to_string(),
            "Korea".to_string(),
            "docid111".to_string(),
            "20240309".to_string(),
            "this is kind".to_string(),
        ),
        PatentRecord::new(
            "title 222".to_string(),
            "this is abstract 222".to_string(),
            "Korea".to_string(),
            "docid222".to_string(),
            "20240309".to_string(),
            "this is kind".to_string(),
        ),
        PatentRecord::new(
            "title 333".to_string(),
            "this is abstract 333".to_string(),
            "Korea".to_string(),
            "docid333".to_string(),
            "20240309".to_string(),
            "this is kind".to_string(),
        ),
        PatentRecord::new(
            "title 444".to_string(),
            "this is abstract 444".to_string(),
            "Korea".to_string(),
            "docid444".to_string(),
            "20240309".to_string(),
            "this is kind".to_string(),
        ),
        PatentRecord::new(
            "title 555".to_string(),
            "this is abstract 555".to_string(),
            "Korea".to_string(),
            "docid555".to_string(),
            "20240309".to_string(),
            "this is kind".to_string(),
        ),
    ];

    let embeddings = vec![
        vec![0.1, 0.1, 0.1, 0.1, 0.1],
        vec![0.2, 0.2, 0.2, 0.2, 0.2],
        vec![0.3, 0.3, 0.3, 0.3, 0.3],
        vec![0.4, 0.4, 0.4, 0.4, 0.5],
        vec![0.5, 0.5, 0.5, 0.5, 0.5],
    ];

    if vectordb.client.has_collection(collection_name).await? {
        vectordb.client.delete_collection(collection_name).await?;
    }
    vectordb.create_collection(collection_name, vector_dim).await?;
    let collections_list = vectordb.client.list_collections().await?;
    println!("collections_list: {:?}", collections_list);
    vectordb
        .upsert_embedding_batch(collection_name, &patent_records, &embeddings, 2)
        .await?;
    let count = vectordb
        .client
        .count(&CountPoints {
            collection_name: collection_name.to_string(),
            filter: None,
            exact: None,
            read_consistency: None,
            shard_key_selector: None,
        })
        .await?;
    println!("count: {:?}", count);

    let search_res = vectordb
        .search(collection_name, &vec![0.1, 0.1, 0.1, 0.1, 0.1], 1, Some(0.8f32), None)
        .await?;
    println!("{:?}", search_res);
    Ok(())
}
