use std::fs;
use std::fs::File;
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};

use quick_xml::events::{BytesStart, Event};
use quick_xml::reader::Reader;
use quick_xml::Writer;

use anyhow::{Error, Result};
use chrono::{Datelike, NaiveDate, Weekday};
use rayon::prelude::*;
use regex::Regex;
use reqwest;
use serde::{Deserialize, Serialize};
use tokio;
use tokio::io::AsyncWriteExt;
use zip::read::ZipArchive;

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct PatentRecord {
    pub abstracts: String,
    pub country: String,
    pub docid: String,
    pub publication_date: String,
    pub kind: String,
}

impl PatentRecord {
    pub fn new(abstracts: String, country: String, docid: String, publication_date: String, kind: String) -> Self {
        PatentRecord {
            abstracts,
            country,
            docid,
            publication_date,
            kind,
        }
    }
    pub fn is_full(&self) -> bool {
        if !self.abstracts.is_empty()
            & !self.country.is_empty()
            & !self.docid.is_empty()
            & !self.publication_date.is_empty()
            & !self.kind.is_empty()
        {
            true
        } else {
            false
        }
    }
}

pub fn parse_xml(path_to_ipa: PathBuf) -> Result<Vec<PatentRecord>> {
    let mut reader = Reader::from_file(path_to_ipa).unwrap();
    reader.trim_text(true);

    let mut buf = Vec::new();
    let mut patents: Vec<PatentRecord> = Vec::new();
    let mut junk_buf: Vec<u8> = Vec::new();
    let mut count = 0;
    let mut patent_application_start_flag = false;
    let mut patent_record: Option<PatentRecord> = None;

    println!("--Parsing xml..");
    // streaming code
    loop {
        if patent_application_start_flag & patent_record.as_ref().is_none() {
            patent_application_start_flag = false;
            patent_record = Some(PatentRecord::default());
        };

        match reader.read_event_into(&mut buf) {
            Err(e) => panic!("Error at position {}: {:?}", reader.buffer_position(), e),
            Ok(Event::Eof) => break,
            Ok(Event::Start(e)) => {
                match e.name().as_ref() {
                    b"us-patent-application" => {
                        patent_application_start_flag = true;
                    },

                    b"abstract" => {
                        // parse all the sub-tags under the abstract in to single string.
                        let abstract_bytes = read_to_end_into_buffer(&mut reader, &e, &mut junk_buf).unwrap();

                        let abstract_str = std::str::from_utf8(&abstract_bytes).unwrap();
                        let abstracts = remove_tags(abstract_str, " ");

                        if patent_record.is_some() {
                            patent_record.as_mut().unwrap().abstracts = abstracts;
                        }
                    },

                    b"publication-reference" => {
                        let publication_refs = read_to_end_into_buffer(&mut reader, &e, &mut junk_buf).unwrap();

                        let publication_refs_str = std::str::from_utf8(&publication_refs).unwrap();
                        let publication_refs: String = remove_tags(publication_refs_str, " ");
                        let mut country_docid_kind: Vec<String> =
                            publication_refs.split(" ").map(|s| s.to_string()).collect();
                        country_docid_kind.retain(|x| !x.is_empty());

                        if patent_record.is_some() {
                            patent_record.as_mut().unwrap().country = country_docid_kind.get(0).unwrap().to_string();
                            patent_record.as_mut().unwrap().docid = country_docid_kind.get(1).unwrap().to_string();
                            patent_record.as_mut().unwrap().publication_date =
                                country_docid_kind.get(2).unwrap().to_string();
                            patent_record.as_mut().unwrap().kind = country_docid_kind.get(3).unwrap().to_string();
                        }
                    },
                    _ => (),
                }
            },

            Ok(Event::End(e)) => match e.name().as_ref() {
                b"us-patent-application" => {
                    count += 1;
                    patents.push(patent_record.unwrap());
                    patent_application_start_flag = false;
                    patent_record = None;
                },
                _ => (),
            },

            _ => (),
        }
        buf.clear();
    }
    println!("parsed {count} entities");
    Ok(patents)
}
fn read_to_end_into_buffer<R: BufRead>(
    reader: &mut Reader<R>,
    start_tag: &BytesStart,
    junk_buf: &mut Vec<u8>,
) -> Result<Vec<u8>, quick_xml::Error> {
    let mut depth = 0;
    let mut output_buf: Vec<u8> = Vec::new();
    let mut w = Writer::new(&mut output_buf);
    let tag_name = start_tag.name();
    w.write_event(Event::Start(start_tag.clone()))?;
    loop {
        junk_buf.clear();
        let event = reader.read_event_into(junk_buf)?;
        w.write_event(&event)?;

        match event {
            Event::Start(e) if e.name() == tag_name => depth += 1,
            Event::End(e) if e.name() == tag_name => {
                if depth == 0 {
                    return Ok(output_buf);
                }
                depth -= 1;
            },
            Event::Eof => {
                panic!("read_to_end_into_buffer meets EOF")
            },
            _ => {},
        }
    }
}

fn remove_tags(input: &str, rep: &str) -> String {
    let re = Regex::new(r"<[^>]*>").unwrap();
    let mut result = re.replace_all(input, rep).to_string();
    result.trim().to_string()
}

pub async fn download_weekly_fulltext(
    uspto_url: &str,
    uspto_year: &str,
    save_dir: &str,
    today_utc: &NaiveDate,
) -> Result<PathBuf, Error> {
    let last_thursday_date = find_last_thursday(today_utc);
    let download_url = format_uspto_full_path(uspto_url, uspto_year, last_thursday_date);
    let download_url_str = download_url.to_str().unwrap();
    let filename = download_url.file_name().unwrap().to_str().unwrap();
    let ipa_zip_file_path = Path::new(save_dir).join(filename);
    let formatted_date = last_thursday_date.format("%y%m%d").to_string();
    let ipa_xml_file_path = Path::new(save_dir).join(format!("ipa{formatted_date}.xml"));

    if !ipa_xml_file_path.exists() {
        println!("--there is NO file at: {:?}", ipa_xml_file_path);
        println!("--Start to Download..");

        let response = reqwest::get(download_url_str).await?;
        if response.status().is_success() {
            let bytes = response.bytes().await?;

            let mut file = tokio::fs::File::create(&ipa_zip_file_path).await.unwrap();
            file.write_all(&bytes).await.unwrap();

            println!("File downloaded successfully to {:?}", ipa_zip_file_path);
        } else {
            println!("Failed to download the file. Status: {}", response.status());
        }

        unzip_ipa(save_dir, filename)?;
    } else {
        println!("--Nothing to download, file open from {}", ipa_zip_file_path.display());
    }

    Ok(ipa_xml_file_path)
}

pub fn format_uspto_full_path(uspto_url: &str, uspto_year: &str, last_thursday_date: NaiveDate) -> PathBuf {
    // Format the date as needed
    let mut file_name = "ipa".to_string();
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

pub fn unzip_ipa(path_to_documents: &str, zipfile_name: &str) -> zip::result::ZipResult<()> {
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

pub fn get_abstracts_from_patents<'a>(patents: &'a [PatentRecord]) -> Result<Vec<&'a str>> {
    let abstracts: Vec<&str> = patents
        .iter()
        .filter_map(|record| {
            // Filter out records that are not full and return a reference to the abstracts
            if record.is_full() {
                Some(record.abstracts.as_str())
            } else {
                None
            }
        })
        .collect();
    Ok(abstracts)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_remove_tags() {
        // simple case
        let xml = "<abstract id=\"abstract\"><p id=\"p-0001\" num=\"0000\">contents in the abstract</p></abstract>";
        let result = remove_tags(xml, " ");
        assert_eq!(result, "contents in the abstract");

        // nested case
        let xml = "<abstract id=\"abstract\"><p id=\"p-0001\" num=\"0000\"><chemistry>contents chem</chemistry>contents in the abstract</p></abstract>";
        let result = remove_tags(xml, " ");
        assert_eq!(result, "contents chem contents in the abstract");
    }
}
