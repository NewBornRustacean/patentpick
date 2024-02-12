use std::path::{Path, PathBuf};

use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use anyhow::{Result, Error};
use reqwest;
use chrono::{Datelike, Weekday, NaiveDate};

pub async fn download_weekly_fulltext(uspto_url:&str, uspto_year:&str, save_dir:&str, today_utc: &NaiveDate)->Result<(), Error>{
    let last_thursday_date = find_last_thursday(today_utc);
    let download_url = format_uspto_full_path(uspto_url, uspto_year, last_thursday_date);
    let download_url_str = download_url.to_str().unwrap();
    let filename = download_url.file_name().unwrap().to_str().unwrap();
    let ipa_file_path = Path::new(save_dir).join(filename);

    if !ipa_file_path.exists(){
        println!("--there is NO file at: {:?}", ipa_file_path);
        println!("--Start to Download..");

        let response = reqwest::get(download_url_str).await?;
        if response.status().is_success() {
            let bytes = response.bytes().await?;

            let mut file = File::create(&ipa_file_path).await.unwrap();
            file.write_all(&bytes).await.unwrap();

            println!("File downloaded successfully to {:?}", ipa_file_path);
        } else {
            println!("Failed to download the file. Status: {}", response.status());
        }
    }

    Ok(())
}

pub fn format_uspto_full_path(uspto_url:&str, uspto_year:&str, last_thursday_date:NaiveDate)->PathBuf{
    // Format the date as needed
    let mut file_name ="ipa".to_string();
    let formatted_date = last_thursday_date.format("%y%m%d").to_string();
    file_name.push_str(&formatted_date);
    file_name.push_str(".zip");

    // Construct the full URL path
    let mut full_path = PathBuf::from(&uspto_url);
    full_path.push(uspto_year);
    full_path.push(file_name);

    return full_path;
}

pub fn find_last_thursday(today: &NaiveDate) -> NaiveDate {
    let mut current_date = *today;
    loop {
        if current_date.weekday() == Weekday::Thu {
            break;
        }
        current_date = current_date.pred_opt().unwrap();
    }
    current_date
}

pub fn parse_xml(){}


