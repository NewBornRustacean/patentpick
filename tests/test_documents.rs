use std::path::Path;

use chrono::{NaiveDate, Utc};
use patentpick::documents::{
    download_weekly_fulltext, find_last_thursday, format_uspto_full_path, get_abstracts_from_patents, parse_xml,
    PatentRecord,
};
use tokio;

#[tokio::test]
#[ignore]
#[cfg(not(feature = "exclude_from_ci"))]
async fn test_download_weekly_fulltext() {
    let uspto_url = "https://bulkdata.uspto.gov/data/patent/application/redbook/fulltext";
    let uspto_year = "2024";
    let now_utc = Utc::now();
    let today_utc = now_utc.date_naive();

    let save_dir = "tests/resources/documents";

    let result = download_weekly_fulltext(uspto_url, uspto_year, save_dir, &today_utc).await;
    println!("get results: {:?}", result);
    assert!(result.is_ok());
}

#[test]
fn test_format_uspto_full_path() {
    // Define the input parameters
    let uspto_url = "https://bulkdata.uspto.gov/data/patent/grant/redbook/fulltext/";
    let uspto_year = "2024/";
    let last_thursday = NaiveDate::from_ymd_opt(2024, 2, 8).unwrap();

    // Call the function and store the result
    let result = format_uspto_full_path(uspto_url, uspto_year, last_thursday);

    // Define the expected output
    let expected = "https://bulkdata.uspto.gov/data/patent/grant/redbook/fulltext/2024/ipa240208.zip";

    println!("result_uspto_url: {:?}", result);

    // Assert that the result matches the expected output
    assert_eq!(result.to_str().unwrap().to_string(), expected);
}

#[test]
#[cfg(not(feature = "exclude_from_ci"))]
fn test_parse_xml() {
    let file_path = Path::new("resources/documents/").join("ipa240215.xml");

    let results = parse_xml(file_path).unwrap();
    println!("{:?}", results.get(0));
}

#[test]
fn test_find_last_thursday() {
    // Define some sample dates and their expected last Thursdays
    let samples = [
        (
            NaiveDate::from_ymd_opt(2024, 2, 12),
            NaiveDate::from_ymd_opt(2024, 2, 8),
        ),
        (NaiveDate::from_ymd_opt(2024, 2, 8), NaiveDate::from_ymd_opt(2024, 2, 8)),
        (NaiveDate::from_ymd_opt(2024, 2, 7), NaiveDate::from_ymd_opt(2024, 2, 1)),
        (
            NaiveDate::from_ymd_opt(2024, 1, 1),
            NaiveDate::from_ymd_opt(2023, 12, 28),
        ),
    ];

    // Loop through the samples and assert the results
    for (today, expected) in samples.iter() {
        let result = find_last_thursday(&today.unwrap());
        assert_eq!(result, expected.unwrap());
    }
}

#[test]
fn test_get_abstracts_from_patents() {
    let patents = vec![
        PatentRecord {
            abstracts: "Abstract 1".to_string(),
            country: "Country 1".to_string(),
            docid: "DocID 1".to_string(),
            publication_date: "2024-02-24".to_string(),
            kind: "Kind 1".to_string(),
        },
        PatentRecord {
            abstracts: "Abstract 2".to_string(),
            country: "Country 2".to_string(),
            docid: "DocID 2".to_string(),
            publication_date: "2024-02-25".to_string(),
            kind: "Kind 2".to_string(),
        },
        PatentRecord {
            abstracts: "Abstract 3".to_string(),
            country: "Country 3".to_string(),
            docid: "DocID 3".to_string(),
            publication_date: "2024-02-26".to_string(),
            kind: "Kind 3".to_string(),
        },
    ];

    let result = get_abstracts_from_patents(&patents).unwrap();

    assert_eq!(result, vec!["Abstract 1", "Abstract 2", "Abstract 3"]);
}
