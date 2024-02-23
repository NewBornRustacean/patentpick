use std::env;
use std::path::PathBuf;

use anyhow::{Error, Result};
use lettre::transport::smtp::authentication::Credentials;
use lettre::{
    message::{header, MultiPart, SinglePart},
    Message, SmtpTransport, Transport,
};
use maud::html;
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub enum EmailError {
    HtmlNotComposed,
    SendFailed,
}
pub struct PatentApplicationContent {
    pub title: String,
    pub hyperlink: String,
}

impl PatentApplicationContent {
    pub fn new(title: String, hyperlink: String) -> Self {
        Self { title, hyperlink }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Subscriber {
    pub name: String,
    pub email: String,
    pub search_queries: Vec<String>,
    pub html_to_send: Option<String>,
}

impl Subscriber {
    pub fn new(
        name: String,
        email: String,
        search_queries: Vec<String>,
        html_to_send: Option<String>,
    ) -> Self {
        Subscriber {
            name,
            email,
            search_queries,
            html_to_send,
        }
    }

    pub fn compose_html(&mut self, applications: &Vec<PatentApplicationContent>) -> &Self {
        // Create the html we want to send.
        let html = html! {
            head {
                title { "Hi there! This is weekly PatentPick :)" }
                style type="text/css" {
                    "h2, h4 { font-family: Arial, Helvetica, sans-serif; }"
                }
            }
            table align="center" {
                tr {
                    td {
                        h2 { "Hi there! This is weekly PatentPick :) " }

                        // Substitute in the name of our recipient.
                        p { "Dear " (self.name) ", here is a list of patent applications that you might be interested in. " }
                        @for application in applications {
                            p {
                                a href=(application.hyperlink.clone()) { (application.title.clone()) }
                            }
                        }
                    }
                }
            }
        };
        self.html_to_send = Some(html.into_string());
        self
    }

    pub fn send_email(&self) -> Result<(), EmailError> {
        match &self.html_to_send {
            Some(html_to_send) => {
                let email_from: &str = "huiseomkim@gmail.com";
                let brevo_key = env::var("BREVO_KEY").unwrap();
                let host: &str = "smtp-relay.sendinblue.com";

                let email = Message::builder()
                    .from(email_from.parse().unwrap())
                    .to(self.email.parse().unwrap())
                    .subject("Hello from PatentPick!")
                    .multipart(
                        MultiPart::alternative() // This is composed of two parts.
                            .singlepart(
                                SinglePart::builder()
                                    .header(header::ContentType::TEXT_PLAIN)
                                    .body(String::from(
                                        "Hello from Lettre! A mailer library for Rust",
                                    )), // Every message should have a plain text fallback.
                            )
                            .singlepart(
                                SinglePart::builder()
                                    .header(header::ContentType::TEXT_HTML)
                                    .body(html_to_send.clone()),
                            ),
                    )
                    .expect("failed to build email");

                let mailer: SmtpTransport = SmtpTransport::relay(&host)
                    .unwrap()
                    .credentials(Credentials::new(
                        email_from.to_string(),
                        brevo_key.to_string(),
                    ))
                    .build();

                match mailer.send(&email) {
                    Ok(_) => {
                        println!("your email sent properly!");
                        Ok(())
                    }
                    Err(e) => {
                        println!("couldn't send email {:?}", e);
                        Err(EmailError::SendFailed)
                    }
                }
            }
            None => Err(EmailError::HtmlNotComposed),
        }
    }
}

pub fn get_subscribers(json_path: PathBuf) -> Result<Vec<Subscriber>, Error> {
    let file = std::fs::File::open(json_path)?;
    let subscribers: Vec<Subscriber> = serde_json::from_reader(file)?;

    Ok(subscribers)
}
