CREATE TABLE IF NOT EXISTS wiki_articles (
    id TEXT PRIMARY KEY,
    title TEXT,
    text TEXT,
    length INT,
    created_at TIMESTAMP DEFAULT NOW()
);
