use std::fs;
use patentpick::emails::{PatentApplicationContent, Subscriber};


#[test]
fn test_compose_html() {
    let mut subscriber_seollem = Subscriber::new("Seollem".to_string(), "email_to_selloem@gmail.com".to_string(), None);

    let applications = vec![
        PatentApplicationContent {
            title: "Patent 1".to_string(),
            hyperlink: "http://example.com/patent1".to_string(),
        },
        PatentApplicationContent {
            title: "Patent 2".to_string(),
            hyperlink: "http://example.com/patent2".to_string(),
        },
    ];

    subscriber_seollem.compose_html(&applications);
    let html = subscriber_seollem.html_to_send.unwrap();
    fs::write("test.html",html.into_string()).expect("unable to write file");
}
