use reqwest;
use chrono::{Datelike, Utc, Weekday};
use patentpick::settings;


pub fn download_weekly_fulltext() {
    let now_utc = Utc::now();
    let today_utc = now_utc.date_naive();
    let thursday = today_utc.with_weekday(Weekday::Thu);

    // Format the date as YYMMDD
    let date_str = format!("{:02}{:02}{:02}", thursday.year() % 100, thursday.month(), thursday.day());

    // Construct the file name and the download URL
    // let file_name = format!("ipa{}.zip", date_str);
    // let download_url = format!("{}{}", BASE_URL, file_name);
    //
    // // Send a blocking GET request to the download URL
    // let resp = reqwest::blocking::get(&download_url).expect("request failed");
    //
    // // Create the directory to save the file if it does not exist
    // std::fs::create_dir_all(SAVE_DIR).expect("failed to create directory");
    //
    // // Create the file in the directory
    // let mut out = File::create(format!("{}/{}", SAVE_DIR, file_name)).expect("failed to create file");
    //
    // // Copy the response body to the file
    // copy(&mut resp.bytes().expect("failed to read bytes").as_ref(), &mut out).expect("failed to copy content");
}


pub fn parse_xml(){}