use std::env;

use lettre::transport::smtp::authentication::Credentials;
use lettre::{Message, SmtpTransport, Transport};

fn main() {
    let brevo_key = env::var("BREVO_KEY").unwrap();
    let email_from:&str = "huiseomkim@gmail.com";
    let host:&str = "smtp-relay.sendinblue.com";
    let email_to :&str = "huiseomkim@gmail.com";

    let email:Message = Message::builder()
        .from(email_from.parse().unwrap())
        .to(email_to.parse().unwrap())
        .subject("test to send an email")
        .body("Hello world! go go patent pick!".to_string())
        .unwrap();

    let mailer:SmtpTransport = SmtpTransport::relay(&host)
        .unwrap()
        .credentials(Credentials::new(
            email_from.to_string(), brevo_key.to_string(),
        ))
        .build();

    match mailer.send(&email) {
        Ok(_) => println!("your email sent properly!"),
        Err(e) => println!("couldn't send email {:?}", e),
    }

}