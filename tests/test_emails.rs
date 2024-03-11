use patentpick::emails::{get_subscribers, PatentApplicationContent, Subscriber};
use qdrant_client::prelude::Payload;
use std::fs;
use std::io::Write;
use std::path::Path;

use serde_json::json;

#[test]
#[cfg(not(feature = "exclude_from_ci"))]
fn test_compose_html() {
    let mut subscriber_seollem = Subscriber::new(
        "Seollem".to_string(),
        "email_to_selloem@gmail.com".to_string(),
        vec!["glp-1 inhibitor".to_string()],
        None,
    );

    let applications = vec![
        PatentApplicationContent {
            title: "Patent 1".to_string(),
            application_abstract: "this is abstract from patent 1".to_string(),
            link_to_pdf: "https://ppubs.uspto.gov/dirsearch-public/print/downloadPdf/20240049614".to_string(),
        },
        PatentApplicationContent {
            title: "Patent 2".to_string(),
            application_abstract: "this is abstract from patent 2".to_string(),
            link_to_pdf: "https://ppubs.uspto.gov/dirsearch-public/print/downloadPdf/20240049615".to_string(),
        },
    ];

    subscriber_seollem.compose_html(&applications);
    let html = subscriber_seollem.html_to_send.unwrap();
    fs::write("test.html", html).expect("unable to write file");
}

#[test]
#[cfg(not(feature = "exclude_from_ci"))]
fn test_get_subscribers() {
    let json_path = Path::new(&"resources/subscribers/".to_string()).join("subscribers.json");
    let subscribers = get_subscribers(json_path).unwrap();

    for subs in subscribers {
        println!("{:?}", subs);
    }
}

#[test]
fn test_from_payload() {
    let payload: Payload = json!({
        "title": "title 111",
        "abstracts": "this is abstract 111",
        "country": "Korean",
        "docid": "docid111",
        "publication_date": "240311",
        "kind": "A1",
    })
    .try_into()
    .unwrap();
    let uspto_pdf_url = "https://ppubs.uspto.gov/dirsearch-public/print/downloadPdf/";

    let content = PatentApplicationContent::from_payload(&payload, uspto_pdf_url);

    assert_eq!(content.title, "title 111");
    assert_eq!(content.application_abstract, "this is abstract 111");
    assert_eq!(content.link_to_pdf, "https://ppubs.uspto.gov/dirsearch-public/print/downloadPdf/docid111");
}
