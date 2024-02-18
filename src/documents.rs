use std::path::{Path, PathBuf};
use std::fs;
use std::fs::File;
use std::io::{BufRead, Read, BufReader};

use quick_xml::de;
use quick_xml::events::Event;
use quick_xml::reader::Reader;

use serde::{Deserialize, Serialize};

use zip::read::ZipArchive;
use tokio;
use tokio::io::AsyncWriteExt;
use anyhow::{Result, Error};
use reqwest;
use chrono::{Datelike, Weekday, NaiveDate};

#[derive(Debug, Default)]
pub struct PatentRecord{
    pub abstracts: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct Ptag{
    #[serde(rename="$value")]
    content: String,
}

// pub fn parse_xml(path_to_ipazip:PathBuf) {
//     let zipfile = File::open(path_to_ipazip).unwrap();
//     let mut archive = ZipArchive::new(zipfile).unwrap();
//
//     // Assuming there is only one XML file in the zip
//     let mut xml_file = archive.by_index(0).unwrap();
//
//     let mut reader= Reader::from_reader((BufReader::new(&mut xml_file)));
//     // reader.config_mut().trim_text(true);
//     reader.trim_text(true);
//
//     let mut buf = Vec::new();
//     let mut abstract_section = PatentRecord::default();
//     let mut in_abstract = false;
//
//     while let Ok(evt) = reader.read_event_into(&mut buf) {
//         match evt {
//             Event::Start(ref e) => {
//                 match e.name() {
//                     b"abstract" => in_abstract = true,
//                     b"p" if in_abstract => {
//                         // Assuming the content of <p> is simple text without nested tags
//                         if let Ok(text) = reader.read_text(e.name(), &mut Vec::new()) {
//                             abstract_section.abstracts.push(text);
//                         }
//                     },
//                     _ => (), // Handle other tags like <chemistry> if needed
//                 }
//             },
//             Event::End(ref e) => {
//                 if e.name() == b"abstract" {
//                     in_abstract = false; // Exiting the <abstract> section
//                 }
//             },
//             Event::Eof => break,
//             _ => (),
//         }
//         buf.clear();
//     }
//
//     println!("{:?}", abstract_section);
//
// }

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

            let mut file = tokio::fs::File::create(&ipa_file_path).await.unwrap();
            file.write_all(&bytes).await.unwrap();

            println!("File downloaded successfully to {:?}", ipa_file_path);
        } else {
            println!("Failed to download the file. Status: {}", response.status());
        }

        unzip_ipa(save_dir, filename)?;
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


pub fn unzip_ipa(path_to_documents:&str, zipfile_name:&str) -> zip::result::ZipResult<()> {
    let path_docs = Path::new(path_to_documents);
    let ipa_zip_path = path_docs.join(zipfile_name);
    let file_open_zip = File::open(&ipa_zip_path)?;

    let mut archive = ZipArchive::new(file_open_zip)?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = path_docs.join(file.mangled_name());

        if (&*file.name()).ends_with('/') {
            println!("File {} extracted to \"{}\"", i, outpath.display());
            fs::create_dir_all(&outpath)?;
        } else {
            println!("File {} extracted to \"{}\" ({} bytes)", i, outpath.display(), file.size());
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(&p)?;
                }
            }
            let mut outfile = fs::File::create(&outpath)?;
            std::io::copy(&mut file, &mut outfile)?;
        }
    }

    // Remove the zip file
    fs::remove_file(&ipa_zip_path)?;
    println!("zip file removed:{}", ipa_zip_path.display());

    Ok(())
}
