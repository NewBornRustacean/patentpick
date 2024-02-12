use std::path::Path;
use tokio;
use chrono::{NaiveDate, Utc};
use patentpick::documents::{download_weekly_fulltext, find_last_thursday, format_uspto_full_path};

# [tokio::test]
# [ignore]
#[cfg(not(feature = "exclude_from_ci"))]
async fn test_download_weekly_fulltext(){
    let uspto_url = "https://bulkdata.uspto.gov/data/patent/application/redbook/fulltext";
    let uspto_year = "2024";
    let now_utc = Utc::now();
    let today_utc = now_utc.date_naive();

    let save_dir="tests/resources/documents";

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
fn test_find_last_thursday() {
    // Define some sample dates and their expected last Thursdays
    let samples = [
        (NaiveDate::from_ymd_opt(2024, 2, 12), NaiveDate::from_ymd_opt(2024, 2, 8)),
        (NaiveDate::from_ymd_opt(2024, 2, 8), NaiveDate::from_ymd_opt(2024, 2, 8)),
        (NaiveDate::from_ymd_opt(2024, 2, 7), NaiveDate::from_ymd_opt(2024, 2, 1)),
        (NaiveDate::from_ymd_opt(2024, 1, 1), NaiveDate::from_ymd_opt(2023, 12, 28)),
    ];

    // Loop through the samples and assert the results
    for (today, expected) in samples.iter() {
        let result = find_last_thursday(&today.unwrap());
        assert_eq!(result, expected.unwrap());
    }
}