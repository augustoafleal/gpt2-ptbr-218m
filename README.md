# LLM Project

Projeto modular para treino de um modelo de linguagem estilo GPT, começando pelo módulo de ingestão de dados da Wikipedia para PostgreSQL.

## Estrutura do repositório

Apenas arquivos e diretórios versionados pelo Git estão listados abaixo.

```text
llm-project/
├── ingest/
│   ├── __init__.py
│   ├── config.py
│   ├── db.py
│   └── ingest.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataloader.py
│   │   └── sft_dataset.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── generate.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── gpt.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── sft_trainer.py
│   │   └── trainer.py
│   └── utils/
│       └── __init__.py
├── scripts/
│   ├── aggregate_judge_results.py
│   ├── audit_benchmark.py
│   ├── build_judge_inputs.py
│   ├── chat_hf.py
│   ├── download_sft_dataset.py
│   ├── download_wiki.py
│   ├── export_extracted_to_training_data.py
│   ├── export_huggingface.py
│   ├── export_training_data.py
│   ├── extract_wiki.py
│   ├── find_best_question.py
│   ├── generate_text.py
│   ├── plot_metrics.py
│   ├── plot_paper_metrics.py
│   ├── prepare_sft_dataset.py
│   ├── prepare_sft_response_only.py
│   ├── run_inference_suite.sh
│   ├── run_local_generation_benchmark.py
│   ├── tokenize_dataset.py
│   ├── train_gpt.py
│   ├── train_sft.py
│   ├── train_tokenizer.py
│   ├── validate_tokenized.py
│   └── validate_tokenizer.py
├── sql/
│   └── init.sql
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Artefatos gerados

Os diretórios abaixo **não são versionados pelo Git**. São criados localmente pelos scripts do pipeline.

| Diretório | Função | Criado por |
|---|---|---|
| `data/` | Dumps, artigos extraídos, corpora de treino, binários tokenizados, datasets SFT | `download_wiki.py`, `extract_wiki.py`, `export_training_data.py`, `tokenize_dataset.py`, `prepare_sft_dataset.py` |
| `runs/` | Checkpoints de treino (`best.pt`, `last.pt`), CSVs de métricas, configurações, saídas de inferência | `train_gpt.py`, `train_sft.py`, `run_inference_suite.sh` |
| `exports/` | Exportações de modelo no formato Hugging Face (`config.json`, `model.safetensors`, arquivos do tokenizer) | `export_huggingface.py` |
| `benchmark/` | Saídas dos modelos, inputs/results dos judges, leaderboard e relatórios de auditoria do benchmark | `run_local_generation_benchmark.py`, `build_judge_inputs.py`, `aggregate_judge_results.py`, `audit_benchmark.py` |
| `artifacts/` | Modelo e vocabulário do tokenizer SentencePiece treinado | `train_tokenizer.py` |

Todos os scripts criam seus diretórios necessários automaticamente — não é preciso `mkdir` manual.

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

## 4a. Exportar dataset diretamente (alternativa sem PostgreSQL)

Rota alternativa que gera o mesmo formato de saída de `export_training_data.py` mas lê
diretamente dos JSON lines do WikiExtractor, **sem passar pelo PostgreSQL**.

Este fluxo bifurca do pipeline principal após a extração:

```text
WikiExtractor                      WikiExtractor
    ↓                                    ↓
data/extracted/             vs.   data/extracted/
    ↓                                    ↓
ingest.py (DB)                          export_extracted_to_training_data.py
    ↓                                    ↓
PostgreSQL                              dataset_general_full.txt
    ↓
export_training_data.py
    ↓
dataset_general.txt
```

```bash
python scripts/export_extracted_to_training_data.py \
    --input-dir data/extracted \
    --output-file data/training/dataset_general_full.txt
```

O script:

- percorre recursivamente o diretório informado em `--input-dir` procurando arquivos `wiki_*`
- processa os JSON lines linha a linha (formato WikiExtractor)
- reusa as mesmas funções de filtro e normalização do pipeline tradicional:
  - `normalize_article()` de `ingest/ingest.py` — valida `id`, `title`, `length >= MIN_TEXT_LENGTH`
  - `clean_field()` de `scripts/export_training_data.py` — remove quebras de linha internas e normaliza espaços
- deduplica artigos por `id` em memória (`set()`)
- ordena por `id` (determinístico, equivalente ao `ORDER BY id` do PostgreSQL)
- escreve no formato `{title}\n{text}\n\n<eos>\n` — **exatamente** o mesmo formato do pipeline DB
- gera metadados em `<output>.metadata.json`

Exemplo do metadado gerado:

```json
{
  "input_dir": "data/extracted",
  "output_file": "data/training/dataset_general_full.txt",
  "total_files": 2906,
  "total_articles_seen": 896342,
  "total_articles_written": 896342,
  "duplicates": 0,
  "skipped_empty": 0,
  "skipped_short": 0,
  "total_characters": 2382480029,
  "min_length": 200,
  "elapsed_seconds": 180.5
}
```

| Argumento | Default | Descrição |
|---|---|---|
| `--input-dir` | `data/extracted` | Diretório com arquivos `wiki_*` do WikiExtractor |
| `--output-file` | `data/training/dataset_general_full.txt` | Arquivo de saída no formato `TITLE\nTEXT\n\n<eos>\n` |
| `--min-length` | `MIN_TEXT_LENGTH` do `.env` (ou `200`) | Comprimento mínimo do texto |

### Diferenças para o pipeline DB

| Característica | Pipeline DB | Pipeline direto |
|---|---|---|
| Dependência | PostgreSQL rodando | Nenhuma |
| Armazenamento intermediário | Tabela no PostgreSQL (~6-7 GB para o corpus completo) | Apenas o arquivo TXT final |
| Deduplicação | `ON CONFLICT (id) DO NOTHING` no banco | `set()` em memória |
| Ordenação | `ORDER BY id` no SQL | `sorted()` em Python |
| Streaming | Streaming do cursor (fetchmany) | **Acumula em memória** para ordenação |
| Tolerância a falhas | Reconexão automática, retry por batch | Falha em erro de leitura de arquivo |

### Limitação conhecida

Devido à ordenação por `id`, o script carrega todos os artigos em memória antes de escrever.
Para o corpus completo (896.342 artigos, ~2,38 GB de texto), o pico de memória é de
aproximadamente **2,5 GB** (texto + overhead Python). Máquinas com menos de 6 GB de RAM podem não conseguir executar a ordenação sem swap.

### Compatibilidade

O arquivo gerado (`dataset_general_full.txt`) é consumido sem modificações por:

- `scripts/tokenize_dataset.py` — treina o tokenizer e gera `train.bin`/`val.bin`
- `scripts/train_gpt.py` — treina o modelo

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

### Retomar treino interrompido

Se o treino for interrompido (energia, OOM, Ctrl+C), é possível retomar do último checkpoint sem perder progresso:

```bash
python scripts/train_gpt.py \
  --resume runs/20260603_220422/last.pt \
  --device cpu \
  --max-iters 100000
```

O checkpoint restaura os pesos do modelo, o estado do optimizer, o step atual e a melhor val loss. O treino continua do step seguinte e o learning rate scheduler (warmup + cosine decay) prossegue exatamente de onde parou. O `--max-iters` define o novo step final — útil para estender o treino além do planejado inicialmente.

### Argumentos do script

| Argumento | Default | Descrição |
|---|---|---|
| `--device` | `cpu` | `cpu` ou `cuda` |
| `--batch-size` | `32` | Tamanho do batch |
| `--block-size` | `256` | Tamanho do contexto |
| `--lr` | `3e-4` | Learning rate |
| `--max-iters` | `10000` | Iterações de treino (define o step final no resume) |
| `--eval-interval` | `500` | Intervalo entre avaliações |
| `--eval-iters` | `100` | Iterações para média da val loss |
| `--n-embd` | `384` | Dimensão do embedding |
| `--n-head` | `6` | Número de cabeças de atenção |
| `--n-layer` | `6` | Número de camadas Transformer |
| `--dropout` | `0.1` | Dropout rate |
| `--resume` | — | Caminho para `last.pt` de treino anterior. Restaura modelo, optimizer, step e best val loss. `--max-iters` define o novo step final. |

## 9. Gerar texto com modelo treinado

Após o treino, gere texto autoregressivamente com um checkpoint salvo.

```bash
python scripts/generate_text.py \
    --checkpoint runs/<timestamp>/best.pt \
    --prompt "Astronomia" \
    --max-new-tokens 200 \
    --temperature 0.8 \
    --top-k 40 \
    --stop-at-eos
```

O script:

- carrega o checkpoint e o `model_config` salvo
- carrega o tokenizer SentencePiece
- codifica o prompt com o tokenizer
- gera token a token usando amostragem com `temperature` e `top_k`
- decodifica os tokens gerados para texto
- opcionalmente interrompe a geração ao encontrar o token `<eos>` (`--stop-at-eos`)

### Argumentos do script

| Argumento | Default | Descrição |
|---|---|---|
| `--checkpoint` | (obrigatório) | Caminho para o `.pt` do checkpoint |
| `--prompt` | `""` | Texto inicial para geração |
| `--max-new-tokens` | `200` | Máximo de tokens a gerar |
| `--temperature` | `0.8` | Temperatura de amostragem |
| `--top-k` | `40` | Top-k amostragem |
| `--device` | `cpu` | `cpu` ou `cuda` |
| `--stop-at-eos` | `false` | Interrompe geração ao emitir `<eos>` |

## 10. Executar inferência em lote

Gera texto para múltiplos prompts pré-definidos a partir de um checkpoint salvo, automatizando o `generate_text.py` em lote.

```bash
./scripts/run_inference_suite.sh <run_id>
```

Exemplo:

```bash
./scripts/run_inference_suite.sh 20260531_232031
```

O script:

- usa o checkpoint `runs/<run_id>/best.pt`
- executa `generate_text.py` para 8 prompts fixos (astronomia, Brasil, IA, história, etc.)
- salva cada saída em `runs/<run_id>/inference/<nome>.txt`
- gera `runs/<run_id>/inference/metadata.json` com os arquivos produzidos

## 11. Visualizar métricas de treino

Gera gráficos de loss e perplexity a partir dos CSVs salvos durante o treino.

```bash
python scripts/plot_metrics.py <run_id>
```

Exemplo:

```bash
python scripts/plot_metrics.py 20260531_232031
```

O script:

- lê `runs/<run_id>/train_metrics.csv` e `runs/<run_id>/eval_metrics.csv`
- gera `train_loss.png` (curva azul `step × loss`) e `val_perplexity.png` (curva laranja `step × perplexity`)
- salva os PNGs dentro do próprio diretório do run

| Argumento | Descrição |
|---|---|
| `--no-grid` | Remove a grade dos gráficos (ativada por padrão) |

## 12. Baixar dataset SFT (Supervised Fine-Tuning)

Baixa o dataset `dominguesm/alpaca-data-pt-br` do Hugging Face para preparação de fine-tuning.

```bash
python scripts/download_sft_dataset.py
```

O script:

- baixa o dataset com `datasets.load_dataset`
- salva em `data/sft/alpaca_ptbr/raw/alpaca_data_ptbr.json`
- preserva os campos `instruction`, `input` e `output`
- é idempotente — se o arquivo já existir, não baixa novamente

| Argumento | Default | Descrição |
|---|---|---|
| `--dataset-name` | `dominguesm/alpaca-data-pt-br` | Nome do dataset no Hugging Face |
| `--output-path` | `data/sft/alpaca_ptbr/raw/alpaca_data_ptbr.json` | Caminho de saída |

## 13. Preparar dataset SFT para treino

Normaliza, formata, divide em train/val, tokeniza com SentencePiece e gera bins uint16 para SFT causal LM.

```bash
python scripts/prepare_sft_dataset.py
```
> Usar --max-examples 1000 para testes rápidos

O script:

- carrega o JSON baixado em `data/sft/alpaca_ptbr/raw/alpaca_data_ptbr.json`
- normaliza exemplos: valida `instruction` e `output`, descarta inválidos, normaliza espaços
- embaralha deterministicamente com seed fixa
- formata cada exemplo no padrão instruction/response com `eos_id` anexado manualmente:

```text
### Instrução:
{instruction}

### Resposta:
{output}
<eos>
```

Quando `input` não está vazio:

```text
### Instrução:
{instruction}

### Entrada:
{input}

### Resposta:
{output}
<eos>
```

- divide em train/val (90/10 por padrão)
- tokeniza com SentencePiece (`artifacts/tokenizer/tokenizer.model`)
- salva `sft_train.txt`, `sft_val.txt`, `train.bin`, `val.bin` (uint16), `metadata.json`

| Argumento | Default | Descrição |
|---|---|---|
| `--input-path` | `data/sft/alpaca_ptbr/raw/alpaca_data_ptbr.json` | JSON de entrada |
| `--output-dir` | `data/sft/alpaca_ptbr/processed` | Diretório de saída |
| `--tokenizer-path` | `artifacts/tokenizer/tokenizer.model` | Tokenizer SentencePiece |
| `--val-ratio` | `0.1` | Proporção de validação |
| `--seed` | `42` | Seed para embaralhamento |
| `--max-examples` | (todos) | Limitar número de exemplos |

Arquivos gerados em `data/sft/alpaca_ptbr/processed/`:

| Arquivo | Descrição |
|---|---|
| `sft_train.txt` | Exemplos de treino formatados |
| `sft_val.txt` | Exemplos de validação formatados |
| `train.bin` | Tokens de treino em `uint16` |
| `val.bin` | Tokens de validação em `uint16` |
| `metadata.json` | Metadados: `vocab_size`, `dtype`, `eos_id`, `train_tokens`, `val_tokens`, etc. |

### 13.1 Preparar dataset SFT Response-Only

Variante do dataset acima que gera máscaras de loss para treinar apenas os tokens da resposta.

```bash
python scripts/prepare_sft_response_only.py
```

Diferenças em relação ao `prepare_sft_dataset.py`:

- gera `train_loss_mask.bin` e `val_loss_mask.bin` (`uint8`: 0 = ignorar loss, 1 = calcular loss)
- metadados incluem `training_format: "response_only"`, `loss_tokens_train/val`, `response_tokens_train/val`
- os tokens de instrução/entrada são mascarados (loss = 0); apenas a resposta e `<eos>` contribuem para a loss

| Argumento | Default | Descrição |
|---|---|---|
| `--input-path` | `data/sft/alpaca_ptbr/raw/alpaca_data_ptbr.json` | JSON de entrada |
| `--output-dir` | `data/sft/alpaca_ptbr/processed_response_only` | Diretório de saída |
| `--tokenizer-path` | `artifacts/tokenizer/tokenizer.model` | Tokenizer SentencePiece |
| `--val-ratio` | `0.1` | Proporção de validação |
| `--seed` | `42` | Seed para embaralhamento |
| `--max-examples` | (todos) | Limitar número de exemplos |

Arquivos gerados em `data/sft/alpaca_ptbr/processed_response_only/`:

| Arquivo | Descrição |
|---|---|
| `train.bin` | Tokens de treino em `uint16` |
| `val.bin` | Tokens de validação em `uint16` |
| `train_loss_mask.bin` | Máscara de loss (treino) em `uint8` |
| `val_loss_mask.bin` | Máscara de loss (validação) em `uint8` |
| `metadata.json` | Metadados com `training_format: "response_only"` |

## 14. Treinar SFT (Supervised Fine-Tuning)

Faz fine-tuning do modelo pré-treinado no dataset Alpaca PT-BR.
Suporta duas modalidades:

- **full_loss** (padrão): loss calculada sobre todos os tokens
- **response_only**: loss calculada apenas sobre os tokens da resposta (instrução/entrada ignorados)

```bash
# Full-loss (padrão)
python scripts/train_sft.py --pretrained-run-id <run_id> --device cuda

# Response-only
python scripts/train_sft.py --pretrained-run-id <run_id> --response-only --device cuda
```

Exemplo com smoke test:

```bash
python scripts/train_sft.py \
  --pretrained-run-id 20260531_232031 \
  --max-iters 100 \
  --batch-size 2 \
  --device cpu
```

Exemplo com treino real (GPU):

```bash
python scripts/train_sft.py \
  --pretrained-run-id 20260531_232031 \
  --batch-size 16 \
  --block-size 256 \
  --max-iters 1000 \
  --eval-interval 50 \
  --eval-iters 20 \
  --lr 5e-5 \
  --min-lr 5e-6 \
  --device cuda
```

O script:

- carrega o checkpoint de `runs/<pretrained_run_id>/best.pt`
- reconstrói o modelo GPT com a mesma configuração do checkpoint base
- mantém **todos os parâmetros treináveis** (full fine-tuning, sem LoRA)
- carrega `train.bin` e `val.bin` do diretório especificado
- com `--response-only`: carrega `processed_response_only/` e usa máscaras para ignorar instrução/entrada
- usa `get_batch` com `np.memmap` (streaming, sem carregar tudo em RAM)
- usa LR menor que o pretraining (default `5e-5`) com warmup + cosine decay
- salva tudo em `runs/sft_<timestamp>/`

| Argumento | Default | Descrição |
|---|---|---|
| `--pretrained-run-id` | (obrigatório) | Run ID do modelo pré-treinado |
| `--checkpoint-name` | `best.pt` | Nome do checkpoint no run base |
| `--data-dir` | `processed` ou `processed_response_only` | Diretório do dataset SFT (auto: `processed` sem flag, `processed_response_only` com `--response-only`) |
| `--batch-size` | `16` | Tamanho do batch |
| `--block-size` | `256` | Tamanho do contexto |
| `--max-iters` | `1000` | Iterações de treino |
| `--eval-interval` | `50` | Intervalo entre avaliações |
| `--eval-iters` | `20` | Iterações para média da val loss |
| `--lr` | `5e-5` | Learning rate |
| `--min-lr` | `5e-6` | Learning rate mínimo |
| `--warmup-iters` | `100` | Iterações de warmup |
| `--lr-decay-iters` | `max_iters` | Iterações para decair o LR |
| `--weight-decay` | `0.1` | Weight decay |
| `--grad-clip` | `1.0` | Gradiente clipping |
| `--device` | auto | `cpu` ou `cuda` |
| `--seed` | `42` | Seed aleatória |
| `--response-only` | `false` | Treina apenas tokens da resposta (ignora instrução/entrada na loss) |

Arquivos gerados em `runs/sft_<timestamp>/`:

| Arquivo | Descrição |
|---|---|
| `best.pt` | Checkpoint com menor val loss |
| `last.pt` | Checkpoint do último passo |
| `config.json` | Configuração completa do treino SFT |
| `run_metadata.json` | Linhagem: checkpoint base, dataset, tipo de treino |
| `train_metrics.csv` | Loss e tokens/sec por step |
| `eval_metrics.csv` | Val loss e perplexity por avaliação |

## 15. Exportar checkpoint para Hugging Face

Exporta um checkpoint treinado (pré-treinamento ou SFT) para o formato Hugging Face,
gerando `config.json`, `model.safetensors`, tokenizer (`tokenizer.model`, `tokenizer_config.json`,
`special_tokens_map.json`), `generation_config.json` e `README.md` — tudo pronto para upload
ao Hub ou uso com `AutoModelForCausalLM`.

```bash
# Exportar checkpoint de pré-treinamento
python scripts/export_huggingface.py \
    --checkpoint runs/<run_id>/best.pt \
    --output exports/huggingface/pretrained_<run_id>

# Exportar checkpoint SFT
python scripts/export_huggingface.py \
    --checkpoint runs/sft_<run_id>/best.pt \
    --output exports/huggingface/sft_<run_id>
```

Para validar a exportação (carrega o modelo exportado com `AutoModelForCausalLM` e compara
encoding do tokenizer com SentencePiece original):

```bash
python scripts/export_huggingface.py \
    --checkpoint runs/sft_<run_id>/best.pt \
    --output exports/huggingface/sft_<run_id> \
    --validate
```

| Flag | Descrição |
|---|---|
| `--checkpoint` | Caminho para o `.pt` ou run ID (ex: `sft_20260610_172000`) |
| `--output` | Diretório de saída dos artefatos |
| `--overwrite` | Sobrescreve diretório de saída existente |
| `--validate` | Pós-exportação: carrega modelo + tokenizer e executa inferência |
| `--verbose` | Log detalhado por tensor |

Arquivos gerados em `exports/huggingface/<run_id>/`:

| Arquivo | Descrição |
|---|---|
| `config.json` | Configuração do modelo (GPT2Config) |
| `model.safetensors` | Pesos exportados em formato seguro |
| `tokenizer.model` | Modelo SentencePiece original |
| `tokenizer_config.json` | Configuração do tokenizer |
| `special_tokens_map.json` | Mapeamento de tokens especiais |
| `tokenizer.json` | Tokenizer completo (vocab + merges) |
| `generation_config.json` | Configuração padrão de geração |
| `README.md` | Model card com pipeline, datasets, citação |

O modelo exportado pode ser carregado diretamente com:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("exports/huggingface/sft_<run_id>")
tokenizer = AutoTokenizer.from_pretrained("exports/huggingface/sft_<run_id>")
```

## 16. Chat interativo com modelo Hugging Face

Após exportar o checkpoint para o formato Hugging Face, é possível conversar
interativamente com o modelo diretamente no terminal.

```bash
# Modo local (carrega o modelo com transformers)
python scripts/chat_hf.py --model exports/huggingface/sft_<run_id>

# Modo API (usa InferenceClient do Hugging Face)
python scripts/chat_hf.py --backend api --model augustoafleal/gpt2-ptbr-218m

# Apenas um turno (não interativo) — útil para testes
python scripts/chat_hf.py --model exports/huggingface/sft_<run_id> \
    --max-new-tokens 120 --temperature 0.7 <<< "O que é Python?"
```

O script oferece dois backends:

| Backend | Descrição |
|---|---|
| `local` (padrão) | Carrega o modelo com `transformers.AutoModelForCausalLM` no dispositivo local |
| `api` | Usa `huggingface_hub.InferenceClient` — não requer GPU, apenas token do HF |

No modo **local**, o modelo é carregado em RAM/VRAM e as respostas são geradas
localmente. No modo **api**, as requisições são enviadas para a infraestrutura
do Hugging Face (gratuito, porém limitado).

O template de instrução segue o mesmo padrão do SFT:

```text
### Instrução:
{user_input}

### Resposta:
```

Para desabilitar o template (útil para modelos base sem fine-tuning):

```bash
python scripts/chat_hf.py --model exports/huggingface/pretrained_<run_id> \
    --no-instruction-template
```

Comandos especiais durante o chat interativo:

| Comando | Ação |
|---|---|
| `exit`, `quit`, `sair` | Encerra a sessão |
| `Ctrl+C` / `Ctrl+D` | Encerra a sessão |

### Argumentos do script

| Argumento | Default | Descrição |
|---|---|---|
| `--backend` | `local` | `local` (transformers) ou `api` (InferenceClient) |
| `--model` | (obrigatório) | Caminho local ou Hub ID (ex: `exports/huggingface/...` ou `augustoafleal/gpt2-ptbr-218m`) |
| `--instruction-template` | `True` | Formata entrada como instrução `### Instrução:\n...\n\n### Resposta:\n` |
| `--max-new-tokens` | `80` | Máximo de tokens a gerar por resposta |
| `--temperature` | `0.7` | Temperatura de amostragem |
| `--top-k` | `40` | Top-k amostragem |
| `--top-p` | `1.0` | Top-p (nucleus) amostragem |
| `--do-sample` | `True` | Usar amostragem estocástica (vs. greedy) |
| `--repetition-penalty` | `1.1` | Penalidade por repetição de tokens |
| `--device` | `auto` | `auto`, `cpu` ou `cuda` (apenas backend local) |
| `--dtype` | `auto` | `auto`, `float32`, `float16` ou `bfloat16` (apenas backend local) |

O token do Hugging Face é resolvido automaticamente na seguinte ordem:

1. variável de ambiente `HF_TOKEN`
2. variável de ambiente `HUGGINGFACE_HUB_TOKEN`
3. token armazenado pelo `huggingface-cli login` (`huggingface_hub.get_token()`)

## 17. Executar benchmark automatizado de geração

Executa 20 perguntas fixas em 2 modos de geração (normal e criativo) e salva todas as respostas em JSON estruturado com campos para avaliação manual.

```bash
# Com path completo do checkpoint
python scripts/run_local_generation_benchmark.py runs/sft_20260607_230617/best.pt

# Ou apenas com o run ID (resolve automaticamente runs/<id>/best.pt)
python scripts/run_local_generation_benchmark.py sft_20260607_230617

# Com output customizado
python scripts/run_local_generation_benchmark.py sft_20260607_230617 benchmark.json
```

Arquivo gerado em `runs/<run_id>/benchmark_<timestamp>.json`:

| Campo | Descrição |
|---|---|
| `checkpoint` | Checkpoint utilizado |
| `modes.normal` | Parâmetros do modo normal (temp=0.3, top_k=20) |
| `modes.creative` | Parâmetros do modo criativo (temp=0.7, top_k=40) |
| `results[].normal.raw_output` | stdout completo da geração |
| `results[].normal.answer` | Resposta extraída (após `### Resposta:`) |
| `results[].manual_eval` | Campos para avaliação manual (pontuação 0-2, notas) |

Exemplo do JSON gerado:

```json
{
  "checkpoint": "runs/sft_20260607_230617/best.pt",
  "num_questions": 20,
  "modes": { "normal": { "temperature": 0.3, "top_k": 20 }, "creative": { ... } },
  "results": [
    {
      "id": 1,
      "question": "O que é inteligência artificial?",
      "normal": { "raw_output": "...", "answer": "...", "exit_code": 0 },
      "creative": { "raw_output": "...", "answer": "...", "exit_code": 0 },
      "manual_eval": {
        "normal_score": null,
        "creative_score": null,
        "notes": "",
        "repetition": null,
        "format_followed": null,
        "factual_error": null
      }
    }
  ]
}
```

## 18. Construir inputs para avaliação por LLM judges

Após gerar os outputs do benchmark, prepare entradas anonimizadas para avaliação cega por
LLM judges (GPT, Gemini, Claude). O script embaralha os modelos em cada pergunta com
aliases aleatórios (A/B/C/D) e gera arquivos JSON + Markdown prontos para o judge.

```bash
python scripts/build_judge_inputs.py
```

O script:

- lê todos os `benchmark/model_outputs/benchmark_run_*.json`
- valida consistência: mesmo número de perguntas e mesmo texto entre modelos
- extrai o campo `answer` (ou `response`, `output`, `generated_text`) de cada resultado
- constrói mapeamento cego `{A, B, C, ...} → model_id` por pergunta (seed=42, shuffle determinístico)
- gera arquivos JSON por pergunta em `benchmark/judge_inputs/`
- gera arquivos Markdown por pergunta em `benchmark/judge_inputs_md/` com prompt de avaliação embutido
- salva o mapeamento em `benchmark/judge_mapping.json`

| Argumento | Default | Descrição |
|---|---|---|
| `--input-dir` | `benchmark/model_outputs` | Diretório com `benchmark_run_*.json` |
| `--output-dir` | `benchmark/judge_inputs` | Saída JSON por pergunta |
| `--output-dir-md` | `benchmark/judge_inputs_md` | Saída Markdown por pergunta |
| `--mapping` | `benchmark/judge_mapping.json` | Caminho do mapeamento aliases → modelos |

O prompt de avaliação incluído nos arquivos Markdown define os critérios:

| Critério | 0 | 1 | 2 |
|---|---|---|---|
| Correctness | incorreta | parcialmente correta | correta |
| Instruction Following | não seguiu | seguiu parcialmente | seguiu completamente |
| Factuality | erros factuais importantes | mistura corretos/incorretos | fatos corretos |
| Conciseness | inadequada | aceitável | objetiva |
| Repetition | repetição excessiva | alguma repetição | sem repetição relevante |
| Overall | ruim | aceitável | excelente |

Cada arquivo Markdown (`benchmark/judge_inputs_md/question_001.md`) contém:

```
# Pergunta 1

## Instrução
{pergunta}

---

## Modelo A
{resposta do modelo A}

---

## Modelo B
{resposta do modelo B}

...

## Avaliação
{prompt completo com formato JSON de saída}
```

Arquivos gerados:

| Arquivo | Descrição |
|---|---|
| `benchmark/judge_inputs/question_*.json` | Entrada JSON por pergunta (campos: question_id, question, answers) |
| `benchmark/judge_inputs_md/question_*.md` | Entrada Markdown por pergunta (pronto para copiar para o judge) |
| `benchmark/judge_mapping.json` | Mapeamento `{pergunta: {alias: model_id}}` para decodificar resultados |

## 19. Agregar resultados dos judges em leaderboard

Processa os resultados das avaliações dos judges (salvos manualmente em
`benchmark/judge_results/`) e gera leaderboard consolidado com métricas
por modelo.

```bash
python scripts/aggregate_judge_results.py
```

O script:

- lê `benchmark/judge_results/{gpt,gemini,claude}/question_*.json` de cada judge
- carrega `benchmark/judge_mapping.json` para decodificar aliases → model_id
- extrai scores normalizados (correctness, instruction_following, factuality, conciseness, repetition, overall)
- extrai ranking, winner e loser por avaliação
- valida consistência: `n_scores == n_rankings == num_questions × num_judges`, `total_wins == total_losses`
- calcula por modelo: overall médio, wins, losses, average_rank
- gera leaderboard ordenado por overall (desc) e wins (desc)

| Argumento | Default | Descrição |
|---|---|---|
| `--judge-results-dir` | `benchmark/judge_results` | Diretório com subpastas `gpt/`, `gemini/`, `claude/` |
| `--mapping` | `benchmark/judge_mapping.json` | Mapeamento aliases → modelos |
| `--reports-dir` | `benchmark/reports` | Diretório de saída dos relatórios |

Arquivos gerados em `benchmark/reports/`:

| Arquivo | Descrição |
|---|---|
| `leaderboard.json` | Leaderboard completo em JSON com todas as métricas |
| `leaderboard.csv` | Leaderboard em CSV (rank, model, overall, wins, losses, average_rank) |
| `leaderboard.md` | Leaderboard em Markdown formatado como tabela |
| `judge_summary.json` | Resumo dos judges (modelo usado, quantidade de perguntas) |
| `question_level_results.json` | Resultados por pergunta (winner/loser por judge) |

Exemplo do leaderboard gerado:

```text
| Rank | Modelo              | Overall | Wins | Losses | Avg Rank |
| ---- | ------------------- | ------: | ---: | -----: | -------: |
| 1    | sft_20260610_172000 |    1.98 |   60 |      0 |     1.00 |
| 2    | sft_20260607_230617 |    1.87 |   56 |      4 |     1.07 |
| 3    | 20260603_224519     |    0.42 |    4 |     56 |     3.55 |
| 4    | 20260531_232031     |    0.18 |    0 |     60 |     4.00 |
```

Saída esperada no console:

```text
Validation checks:
  All checks passed (expected=60)

Total evaluations processed: 60
Judges found:
  gpt: gpt-4o (20 questions)
  gemini: gemini-2.0-flash (20 questions)
  claude: claude-3-5-sonnet-20241022 (20 questions)

Top models by overall:
  1. sft_20260610_172000 — overall=1.98, wins=60, losses=0, avg_rank=1.0
  2. sft_20260607_230617 — overall=1.87, wins=56, losses=4, avg_rank=1.07
  3. 20260603_224519 — overall=0.42, wins=4, losses=56, avg_rank=3.55
  4. 20260531_232031 — overall=0.18, wins=0, losses=60, avg_rank=4.0
```

## 20. Auditar resultados do benchmark

Gera um relatório de auditoria verificando a consistência e coerência dos
resultados dos judges.

```bash
python scripts/audit_benchmark.py
```

O script:

- carrega todos os resultados de `benchmark/judge_results/{gpt,gemini,claude}/`
- carrega `benchmark/judge_mapping.json`
- gera `benchmark/reports/audit_report.md` com as seguintes análises:

| Seção | Análise |
|---|---|
| 1. Distribuição dos Scores | Contagem de notas 0/1/2 por modelo e métrica, com média |
| 2. Winner × Overall | Verifica se o winner tem o maior overall score |
| 3. Winner × Ranking[0] | Verifica se o winner ocupa o topo do ranking |
| 4. Loser × Ranking[-1] | Verifica se o loser ocupa o final do ranking |
| 5. Average Rank | Confirma o cálculo de average_rank |
| 6. Distribuição por Juiz | Média e distribuição de overall por judge |
| 7. Severidade dos Juízes | Comparação da média global vs. cada judge |
| 8. Correlação entre Métricas | Pearson entre overall, average_rank e wins |
| 9. Anomalias no Ranking | Detecta aliases faltantes, extras ou duplicatas |
| 10. Conclusão Final | Resumo da integridade dos dados |

Não requer argumentos de linha de comando.

Saída esperada:

```text
Audit report generated: benchmark/reports/audit_report.md

EXECUTIVE SUMMARY — Benchmark Audit

Key findings:
  1. Aggregator is CORRECT — all numerical validations pass
  2. Judges are severe but internally consistent
  3. Overall and average_rank produce the same ordering
  4. Detected ranking anomaly: Gemini Q10 has 'Delta' instead of 'D'
  5. SFT models clearly dominate the leaderboard
```

## 21. Gerar figuras para o paper

Gera figuras prontas para publicação (PNG + PDF, 300 dpi) com melhor acabamento visual,
utilizando todas as runs de pré-treinamento e SFT já concluídas.

```bash
python scripts/plot_paper_metrics.py <run_id1> [<run_id2> ...]
```

Exemplo com todas as runs:

```bash
python scripts/plot_paper_metrics.py 20260603_224519 sft_20260607_230617 sft_20260610_172000 20260531_232031
```

Figuras geradas por run (no diretório `runs/<run_id>/`):

| Arquivo | Conteúdo |
|---|---|
| `paper_train_loss.png` / `.pdf` | Loss de treino com marcador no menor valor |
| `paper_val_loss.png` / `.pdf` | Loss de validação com marcador no melhor checkpoint |
| `paper_val_perplexity.png` / `.pdf` | Perplexidade de validação com marcador no menor valor |

Figura comparativa (em `runs/`):

| Arquivo | Conteúdo |
|---|---|
| `paper_perplexity_comparison.png` / `.pdf` | Gráfico de barras comparando a melhor perplexidade entre runs |

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
python scripts/export_huggingface.py \
    --checkpoint runs/<run_id>/best.pt \
    --output exports/huggingface/pretrained_<run_id> \
    --validate                              # opcional: exportar para Hugging Face
python scripts/generate_text.py --checkpoint runs/<timestamp>/best.pt --prompt "Astronomia"
python scripts/plot_metrics.py <run_id>          # opcional: visualizar métricas
./scripts/run_inference_suite.sh <run_id>         # opcional: inferência em lote
```

## Pipeline alternativo (sem PostgreSQL)

Rota que pula o banco de dados e gera o corpus direto dos JSON lines do WikiExtractor.

```bash
cp .env.example .env
pip install -r requirements.txt
python scripts/download_wiki.py
python scripts/extract_wiki.py
python scripts/export_extracted_to_training_data.py \
    --input-dir data/extracted \
    --output-file data/training/dataset_general_full.txt
python scripts/tokenize_dataset.py
python scripts/train_gpt.py --device cpu
```

**Atenção**: O `tokenize_dataset.py` atualmente lê de `data/training/dataset_general.txt`.
Para usar o novo corpus, aponte manualmente `DATASET_PATH` no script ou renomeie o arquivo.

## Pipeline SFT (pós-pretraining)

Após o pretraining, prepare o dataset para fine-tuning supervisionado:

```bash
pip install -r requirements.txt
python scripts/download_sft_dataset.py
python scripts/prepare_sft_dataset.py
python scripts/prepare_sft_dataset.py --max-examples 1000
python scripts/prepare_sft_response_only.py
python scripts/train_sft.py --pretrained-run-id <run_id> --device cuda
python scripts/train_sft.py --pretrained-run-id <run_id> --response-only --device cuda
python scripts/export_huggingface.py \
    --checkpoint runs/sft_<run_id>/best.pt \
    --output exports/huggingface/sft_<run_id> \
    --validate                              # opcional: exportar para Hugging Face
python scripts/chat_hf.py \
    --model exports/huggingface/sft_<run_id>    # opcional: chat interativo
```

Após o SFT, avalie os modelos com o pipeline de benchmark automatizado:

```bash
# 1. Gerar respostas para 20 perguntas fixas (2 modos: normal e criativo)
python scripts/run_local_generation_benchmark.py sft_<run_id>

# 2. Construir inputs anonimizados para judges
python scripts/build_judge_inputs.py

# 3. Agregar resultados dos judges em leaderboard
python scripts/aggregate_judge_results.py

# 4. Auditar consistência dos resultados
python scripts/audit_benchmark.py
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
docker compose --profile app run --rm app python scripts/download_sft_dataset.py
docker compose --profile app run --rm app python scripts/prepare_sft_dataset.py
docker compose --profile app run --rm app python scripts/train_sft.py --pretrained-run-id <run_id> --device cpu
docker compose --profile app run --rm app python scripts/export_huggingface.py \
    --checkpoint runs/sft_<run_id>/best.pt \
    --output exports/huggingface/sft_<run_id> \
    --validate                              # opcional: exportar para Hugging Face
docker compose --profile app run --rm app python scripts/chat_hf.py \
    --model exports/huggingface/sft_<run_id>    # opcional: chat interativo
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