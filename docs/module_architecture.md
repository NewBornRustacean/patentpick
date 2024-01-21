# PatentPick Modules and Architecture
초안: 2024. 01. 21 NewBornRustacean

## Modules
### Scheduler
- airflow, crontab etc. scheduled job management
- Crawler: daily
- Mailer: weekly, bi-weekly.
- Retriever: referenced by Mailer(=weekly or bi-weekly)
### Mailer
- send emails: brevo, lettre
- make it prettier
### Crawler
- write into patent documents database
- "search-able" database: simple key-word search, semantic search
### Retriever(Recommender)
- search/recommendation from patent database
## Diagram
