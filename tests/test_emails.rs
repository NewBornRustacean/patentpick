use patentpick::emails::{get_subscribers, PatentApplicationContent, Subscriber};
use std::fs;
use std::io::Write;
use std::path::Path;

#[test]
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
            hyperlink: "http://example.com/patent1".to_string(),
        },
        PatentApplicationContent {
            title: "Patent 2".to_string(),
            hyperlink: "http://example.com/patent2".to_string(),
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
