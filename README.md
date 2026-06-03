# LLM Project

Projeto modular para treino de um modelo de linguagem estilo GPT, começando pelo módulo de ingestão de dados da Wikipedia para PostgreSQL.

## Estrutura

```text
llm-project/
├── ingest/
│   ├── ingest.py
│   ├── db.py
│   └── config.py
├── src/
│   ├── model/
│   │   └── gpt.py
│   ├── data/
│   │   └── dataloader.py
│   ├── training/
│   │   └── trainer.py
│   └── inference/
│       └── generate.py
├── scripts/
│   ├── download_wiki.py
│   ├── extract_wiki.py
│   ├── export_training_data.py
│   ├── train_tokenizer.py
│   ├── validate_tokenizer.py
│   ├── tokenize_dataset.py
│   ├── validate_tokenized.py
│   ├── train_gpt.py
│   └── generate_text.py
├── sql/
│   └── init.sql
├── data/
│   ├── raw/
│   │   └── wiki_dump.xml.bz2
│   ├── extracted/
│   │   └── AA...BD/
│   │       └── wiki_00..wiki_99
│   ├── subset/
│   │   └── AA...AE/
│   │       └── wiki_00..wiki_99
│   ├── training/
│   │   └── dataset_general.txt
│   └── tokenized/
│       ├── train.bin
│       ├── val.bin
│       └── metadata.json
├── runs/
│   └── <timestamp>/
│       ├── best.pt
│       ├── last.pt
│       ├── config.json
│       ├── train_metrics.csv
│       └── eval_metrics.csv
├── artifacts/
│   └── tokenizer/
│       ├── tokenizer.model
│       └── tokenizer.vocab
├── docker-compose.yml
├── .env.example
└── README.md
```

## Pré-requisitos

- Docker e Docker Compose
- Python 3.10+ (observação: `wikiextractor` pode falhar no Python 3.11+ com erro de regex)
- Dependências Python:

```bash
pip install -r requirements.txt
```

Em seguida, instale o PyTorch separadamente conforme seu hardware:

```bash
# CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

> O PyTorch não está no `requirements.txt` porque a URL de instalação varia conforme o hardware (CPU vs CUDA).

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

A tabela `wiki_articles` será criada automaticamente com o script [`sql/init.sql`](sql/init.sql).

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
    --no-templates \
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

## 4. Exportar dataset generalista para treino

Gera um arquivo de texto contínuo (`data/training/dataset_general.txt`) pronto para tokenização BPE/SentencePiece e pretraining GPT-like.

```bash
python scripts/export_training_data.py
```

O script:

- conecta no PostgreSQL (reusa `ingest/db.py`)
- consulta `SELECT title, text FROM wiki_articles WHERE length > 200 ORDER BY id`
- exporta em streaming com `fetchmany(1000)` — sem carregar tudo em memória
- aplica limpeza leve: remove quebras de linha e normaliza espaços
- formata cada artigo como `Título\nTexto\n\n<eos>\n`
- gera `data/training/dataset_general.txt` em UTF-8

Exemplo do formato gerado:

```text
Astronomia
Astronomia é uma ciência natural que estuda corpos celestes...

<eos>
Brasil
Brasil é um país localizado na América do Sul...

<eos>
```

## 5. Treinar tokenizer BPE

Treina um tokenizer BPE (SentencePiece) com o dataset generalista exportado.

```bash
python scripts/train_tokenizer.py
```

O script:

- lê `data/training/dataset_general.txt`
- treina um tokenizer BPE com `vocab_size=16000`
- define special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`
- gera `artifacts/tokenizer/tokenizer.model` e `artifacts/tokenizer/tokenizer.vocab`

Saída esperada:

```text
Iniciando treinamento do tokenizer bpe
Dataset: /caminho/para/data/training/dataset_general.txt
Vocab size: 16000
Artefatos: /caminho/para/artifacts/tokenizer

Tokenizer treinado com sucesso!
Modelo: /caminho/para/artifacts/tokenizer/tokenizer.model
Vocabulário: /caminho/para/artifacts/tokenizer/tokenizer.vocab
```

Arquivos gerados:

| Arquivo | Descrição |
|---|---|
| `artifacts/tokenizer/tokenizer.model` | Modelo binário do SentencePiece |
| `artifacts/tokenizer/tokenizer.vocab` | Vocabulário BPE legível |

## 6. Validar tokenizer

Valida a tokenização com palavras reais extraídas do próprio corpus.

```bash
python scripts/validate_tokenizer.py
```

O script:

- escaneia `data/training/dataset_general.txt` e coleta títulos e termos frequentes
- mostra quantos tokens cada palavra gera e a fragmentação (subwords)
- exibe os special tokens e seus IDs

Saída esperada:

```text
Palavra                      Tokens Fragmentação
----------------------------------------------------------------------
Elvas                           2  ▁El + vas
São                             1  ▁São
Condado                         1  ▁Condado
moeda                           1  ▁moeda
Casa                            1  ▁Casa
Reforma                         1  ▁Reforma
...

Special tokens:
  <eos>        -> id=2, encode=[13501, 2]
  <bos>        -> id=1, encode=[13501, 1]
  <pad>        -> id=3, encode=[13501, 3]
  <unk>        -> id=0, encode=[13501, 0, 12668, 0]
```

## 7. Tokenizar dataset para binários

Converte o corpus textual em token ids binários (`.bin`) para consumo pelo DataLoader e treino GPT-like.

```bash
python scripts/tokenize_dataset.py
```

O script:

- carrega `artifacts/tokenizer/tokenizer.model` (SentencePiece BPE)
- lê `data/training/dataset_general.txt` em streaming (linha a linha)
- detecta artigos separados por `<eos>` textual
- tokeniza cada artigo com `sp.encode()` e insere `EOS_ID` (2) manualmente ao final de cada um
- divide deterministicamente: 90% primeiros artigos → train, 10% → val
- salva como arrays `uint16` contíguos (`vocab_size=16000` cabe perfeitamente)
- gera metadados com contagem de tokens e configuração

Saída esperada:

```text
Tokenizer: /caminho/para/artifacts/tokenizer/tokenizer.model
EOS ID:    2
Dataset:   /caminho/para/data/training/dataset_general.txt

Counting articles...
Total articles: 81050
Train articles: 72945
Val articles:   8105

Tokenizing...
  processed 50000/81050 articles

Total tokens:      105617933
Train tokens:       98729599
Val tokens:          6888334

Saving binary files...
Train:    data/tokenized/train.bin
Val:      data/tokenized/val.bin
Metadata: data/tokenized/metadata.json
```

Arquivos gerados:

| Arquivo | Descrição |
|---|---|
| `data/tokenized/train.bin` | ~189 MB — tokens de treino em `uint16` |
| `data/tokenized/val.bin` | ~13 MB — tokens de validação em `uint16` |
| `data/tokenized/metadata.json` | Metadados: `vocab_size`, `dtype`, `train_tokens`, `val_tokens`, `eos_id`, `tokenizer_path` |

### Validar tokenização

Após gerar os binários, valide a integridade dos arquivos:

```bash
python scripts/validate_tokenized.py
```

O script verifica:

- consistência do `metadata.json` com os binários
- range dos token IDs (todos < `vocab_size`)
- contagem e posição correta dos EOS markers
- roundtrip encode/decode em amostras
- decode legível do primeiro e último artigo
- formato do fluxo `[tokens][EOS][tokens][EOS]...`

## 8. Treinar Mini GPT-like

Treina um Transformer decoder-only (estilo GPT-2) com os binários tokenizados.

```bash
python scripts/train_gpt.py --device cpu
```

Para GPU (se disponível):

```bash
python scripts/train_gpt.py --device cuda
```

O script:

- carrega `data/tokenized/train.bin` e `val.bin` via `numpy.memmap` (sem carregar tudo em RAM)
- instancia o modelo GPT com `vocab_size=16000`, `block_size=256`, `n_embd=384`, `n_head=6`, `n_layer=6`
- usa AdamW com warmup + cosine learning rate decay
- calcula validation loss e perplexity em intervalos regulares
- salva checkpoints em `runs/<timestamp>/`

Arquitetura do modelo (`src/model/gpt.py`):

| Componente | Descrição |
|---|---|
| `token_embedding` | Embedding lookup 16000 → 384 (compartilhado com `lm_head`) |
| `position_embedding` | Embedding posicional aprendido 256 → 384 |
| `CausalSelfAttention` | Atenção multi-head (6 heads) com máscara causal |
| `FeedForward` | MLP 384 → 1536 → 384 com GELU |
| `Block` | LayerNorm → Attention → residual → LayerNorm → FFN → residual |
| `lm_head` | Projeção 384 → 16000 (pesos compartilhados) |

Total de parâmetros: ~16.9M

Saída esperada durante o treino:

```text
Training on cpu
Parameters: 16,869,120
Run dir: /caminho/para/runs/20260528_204405

step      1 | loss 9.7819 | tok/s 1090 | lr 3.00e-06
  └─ val loss 9.7253 | perplexity 16735.00

step   1000 | loss 6.2341 | tok/s 1150 | lr 2.84e-04
  └─ val loss 6.5102 | perplexity 672.34
...

Training complete. Best val loss: 5.2341
Checkpoints saved to /caminho/para/runs/20260528_204405
```

Arquivos gerados em `runs/<timestamp>/`:

| Arquivo | Descrição |
|---|---|
| `best.pt` | Checkpoint com menor val loss (modelo + optimizer) |
| `last.pt` | Checkpoint do último passo |
| `config.json` | Hiperparâmetros do treino |
| `train_metrics.csv` | Loss e tokens/sec por step |
| `eval_metrics.csv` | Val loss e perplexity por avaliação |

### Argumentos do script

| Argumento | Default | Descrição |
|---|---|---|
| `--device` | `cpu` | `cpu` ou `cuda` |
| `--batch-size` | `32` | Tamanho do batch |
| `--block-size` | `256` | Tamanho do contexto |
| `--lr` | `3e-4` | Learning rate |
| `--max-iters` | `10000` | Iterações de treino |
| `--eval-interval` | `500` | Intervalo entre avaliações |
| `--eval-iters` | `100` | Iterações para média da val loss |
| `--n-embd` | `384` | Dimensão do embedding |
| `--n-head` | `6` | Número de cabeças de atenção |
| `--n-layer` | `6` | Número de camadas Transformer |
| `--dropout` | `0.1` | Dropout rate |

## 9. Gerar texto com modelo treinado

Após o treino, gere texto autoregressivamente com um checkpoint salvo.

```bash
python scripts/generate_text.py \
    --checkpoint runs/<timestamp>/best.pt \
    --prompt "Astronomia" \
    --max-new-tokens 200 \
    --temperature 0.8 \
    --top-k 40
```

O script:

- carrega o checkpoint e o `model_config` salvo
- carrega o tokenizer SentencePiece
- codifica o prompt com o tokenizer
- gera token a token usando amostragem com `temperature` e `top_k`
- decodifica os tokens gerados para texto

### Argumentos do script

| Argumento | Default | Descrição |
|---|---|---|
| `--checkpoint` | (obrigatório) | Caminho para o `.pt` do checkpoint |
| `--prompt` | `""` | Texto inicial para geração |
| `--max-new-tokens` | `200` | Máximo de tokens a gerar |
| `--temperature` | `0.8` | Temperatura de amostragem |
| `--top-k` | `40` | Top-k amostragem |
| `--device` | `cpu` | `cpu` ou `cuda` |

## Variáveis de ambiente

Arquivo [`.env.example`](.env.example):

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
python scripts/export_training_data.py
python scripts/train_tokenizer.py
python scripts/validate_tokenizer.py
python scripts/tokenize_dataset.py
python scripts/train_gpt.py --device cpu
python scripts/generate_text.py --checkpoint runs/<timestamp>/best.pt --prompt "Astronomia"
```

## Pipeline (Docker-first)

Esse fluxo roda *tudo* (download, extração e ingestão) dentro do container `app`, junto com o PostgreSQL no Compose.

```bash
cp .env.example .env
docker compose up -d postgres
docker compose --profile app run -d --name llm_download app python scripts/download_wiki.py
docker compose --profile app run -d --name llm_extract app python scripts/extract_wiki.py
docker compose --profile app run -d --name llm_ingest app python ingest/ingest.py
docker compose --profile app run --rm app python scripts/export_training_data.py
docker compose --profile app run --rm app python scripts/train_tokenizer.py
docker compose --profile app run --rm app python scripts/validate_tokenizer.py
docker compose --profile app run --rm app python scripts/tokenize_dataset.py
docker compose --profile app run --rm app python scripts/train_gpt.py --device cpu
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
