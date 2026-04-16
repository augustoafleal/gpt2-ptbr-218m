# LLM Project

Projeto modular para treino de um modelo de linguagem estilo GPT, começando pelo módulo de ingestão de dados da Wikipedia para PostgreSQL.

## Estrutura

```text
llm-project/
├── ingest/
│   ├── ingest.py
│   ├── db.py
│   └── config.py
├── scripts/
│   ├── download_wiki.py
│   └── extract_wiki.py
├── sql/
│   └── init.sql
├── docker-compose.yml
├── .env.example
└── README.md
```

## Pré-requisitos

- Docker e Docker Compose
- Python 3.10+
- Dependências Python:

```bash
pip install -r requirements.txt
```

## Configuração

1. Crie o arquivo de ambiente:

```bash
cp .env.example .env
```

2. Ajuste as credenciais se necessário.

## Subir o PostgreSQL (Docker)

O download e a extração rodam em Python e não precisam de Docker. O PostgreSQL é necessário apenas para a etapa de ingestão.

Suba o banco a partir da raiz do projeto:

```bash
docker compose up -d
```

A tabela `wiki_articles` será criada automaticamente com o script [`sql/init.sql`](/home/guto/dev/python/llm-project/sql/init.sql).

## 1. Baixar o dump da Wikipedia

Por padrão, o script baixa o dump mais recente da Wikipedia em português (`ptwiki`) e salva em `data/raw/wiki_dump.xml.bz2`.

```bash
python scripts/download_wiki.py
```

Para outro idioma/projeto, passe o código desejado:

```bash
python scripts/download_wiki.py enwiki
```

No modo Docker-first, o comando é o mesmo, só rodando dentro do serviço `app`:

```bash
docker compose --profile app run --rm app python scripts/download_wiki.py enwiki
```

## 2. Extrair o dump com WikiExtractor

Por padrão, o script gera saída JSON em `data/extracted/`.

```bash
python scripts/extract_wiki.py
```

Comando executado internamente:

```bash
python -m wikiextractor.WikiExtractor \
    --json \
    --no_templates \
    --min_text_length 200 \
    -o data/extracted \
    data/raw/wiki_dump.xml.bz2
```

## 3. Ingerir os artigos no PostgreSQL

Execute a ingestão a partir da raiz do projeto:

```bash
python ingest/ingest.py
```

O pipeline:

- lê os arquivos JSON do diretório extraído definido em `WIKI_EXTRACTED_DIR`
- extrai `id`, `title` e `text`
- calcula `length`
- ignora textos com menos de 200 caracteres
- faz inserções em lote
- ignora duplicados com `ON CONFLICT (id) DO NOTHING`
- faz `commit` por batch
- protege `rollback()` para conexões inválidas
- tenta reconectar automaticamente em erro de conexão com retry único do batch
- continua a execução mesmo se um batch falhar

## Variáveis de ambiente

Arquivo [`.env.example`](/home/guto/dev/python/llm-project/.env.example):

- `POSTGRES_HOST`: host do PostgreSQL
- `POSTGRES_PORT`: porta do PostgreSQL
- `POSTGRES_DB`: nome do banco
- `POSTGRES_USER`: usuário do banco
- `POSTGRES_PASSWORD`: senha do banco
- `BATCH_SIZE`: tamanho do lote de inserção (default mais seguro: `100`)
- `MIN_TEXT_LENGTH`: tamanho mínimo do texto aceito pelo pipeline
- `WIKI_JSON_GLOB`: padrão de busca dos arquivos extraídos
- `WIKI_EXTRACTED_DIR`: diretório extraído (saída do `WikiExtractor`) usado na ingestão

## Pipeline (Python local + Postgres no Docker)

Esse fluxo roda os scripts Python no seu host e usa o PostgreSQL via Docker Compose.

```bash
cp .env.example .env
docker compose up -d
pip install -r requirements.txt
python scripts/download_wiki.py
python scripts/extract_wiki.py
python ingest/ingest.py
```

## Pipeline (Docker-first)

Esse fluxo roda *tudo* (download, extração e ingestão) dentro do container `app`, junto com o PostgreSQL no Compose.

```bash
cp .env.example .env
docker compose up -d postgres
docker compose --profile app run -d --name llm_download app python scripts/download_wiki.py
docker compose --profile app run -d --name llm_extract app python scripts/extract_wiki.py
docker compose --profile app run -d --name llm_ingest app python ingest/ingest.py
```

Logs/estado:

```bash
docker ps -a | rg "llm_(download|extract|ingest)"
docker logs -f llm_download
```

Para outro idioma/projeto:

```bash
docker compose --profile app run -d --name llm_download app python scripts/download_wiki.py enwiki
```

## Comportamento da ingestão

- `BATCH_SIZE=100` é o default recomendado para evitar statements grandes demais com `execute_values`.
- Se o PostgreSQL cair durante um batch, o script tenta fazer `rollback()` com segurança.
- Se a conexão estiver quebrada, o script fecha a conexão antiga, abre uma nova e tenta o mesmo batch mais uma vez.
- Se o retry também falhar, o batch é marcado como falho e o pipeline continua para o próximo batch.
- Os logs mostram batches processados, batches com erro, reconexão e resumo final da ingestão.

Exemplos de logs esperados:

```text
Using extracted data from /caminho/para/data/extracted
Found 128 files to ingest
Processed batch 1 from wiki_00: 100 rows inserted
Connection error on batch 4 from wiki_00: server closed the connection unexpectedly
Database connection re-established
Retried batch 4 from wiki_00: 100 rows inserted
Finished file /caminho/para/wiki_00: 1200 rows inserted across 12 batches
Ingestion summary: 50000 rows inserted, 520 batches processed, 2 batches failed
```

## Observações

- O PostgreSQL é exposto na porta definida em `.env`.
- O script de ingestão carrega `.env` automaticamente via `python-dotenv`.
- O diretório de extração usado pela ingestão é o definido em `WIKI_EXTRACTED_DIR`.
- O script de download falha em HTTP error e tenta retomar downloads interrompidos via arquivo `.part`.
- O script de extração não reutiliza diretórios antigos.
- Se o container já existir sem a tabela, recrie o volume ou execute manualmente o SQL de inicialização.
