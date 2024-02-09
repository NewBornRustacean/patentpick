mod emails;
mod mpnet;

use emails::{Subscriber, PatentApplicationContent};

fn main() {
    let mut subscriber_seom =Subscriber::new("SeomKim".to_string(), "huiseomkim@gmail.com".to_string(), None);

    let mut mock_results = Vec::new();
    mock_results.push(emails::PatentApplicationContent::new(
        "Rapid transformation of monocot leaf explants".to_string(),
        "https://patents.google.com/patent/US20240002870A1/en?oq=US+20240002870+A1".to_string())
    );

    mock_results.push(emails::PatentApplicationContent::new(
        "PYRIDO[2,3-D]PYRIMIDIN-4-AMINES AS SOS1 INHIBITORS".to_string(),
        "https://patents.google.com/patent/US20230357239A1/en?oq=US+20230357239+A1".to_string())
    );

    mock_results.push(emails::PatentApplicationContent::new(
        "PRIME EDITING GUIDE RNAS, COMPOSITIONS THEREOF, AND METHODS OF USING THE SAME".to_string(),
        "https://patents.google.com/patent/US20230357766A1/en?oq=US+20230357766+A1".to_string())
    );

    mock_results.push(emails::PatentApplicationContent::new(
        "IMAGE SENSOR".to_string(),
        "https://patents.google.com/patent/US20230352510A1/en?oq=US+20230352510+A1".to_string())
    );

    mock_results.push(emails::PatentApplicationContent::new(
        "Expressing Multicast Groups Using Weave Traits".to_string(),
        "https://patents.google.com/patent/US20230336371A1/en?oq=US+20230336371+A1".to_string())
    );

    subscriber_seom.compose_html(&mock_results).send_email().unwrap();
}