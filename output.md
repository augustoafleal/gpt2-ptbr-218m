# Relatório de Validação — Tokenizer BPE

## 1. Resumo Executivo

**PASS WITH WARNINGS** — Score: 95/100

Tokenizer treinado novamente com `data/training/dataset_general.txt` (424 MB, 81.050 artigos, UTF-8). Artefatos regenerados, encode/decode consistente, vocabulário íntegro.

---

## 2. Checklist

### 2.1 Estrutura dos Artefatos

| Arquivo | Tamanho | Timestamp |
|---|---|---|
| `artifacts/tokenizer/tokenizer.model` | 481 KB | 27/05 22:37 |
| `artifacts/tokenizer/tokenizer.vocab` | 220 KB | 27/05 22:37 |

✓ Ambos existem, não vazios, tamanhos plausíveis.

### 2.2 Vocabulário — Special Tokens

| Token | ID | `piece_to_id()` | `encode()` |
|---|---|---|---|
| `<unk>` | 0 | 0 | `[0]` quando isolado |
| `<bos>` | 1 | 1 | `[13501, 1]` (prefixo `▁`) |
| `<eos>` | 2 | 2 | `[13501, 2]` (prefixo `▁`) |
| `<pad>` | 3 | 3 | `[13501, 3]` (prefixo `▁`) |

IDs especiais presentes e corretos. `bos_id()`/`eos_id()`/`pad_id()` retornam `-1` (comportamento normal do SentencePiece com `user_defined_symbols`). Usar `piece_to_id()` no DataLoader.

### 2.3 Tamanho do Vocabulário

`sp.get_piece_size()` = **16000** ✓

### 2.4 Encode/Decode

```
Texto:    "Astronomia é uma ciência natural."
IDs:      [373, 6351, 72, 69, 5195, 2005, 13518]
Decoded:  "Astronomia é uma ciência natural."
Roundtrip: OK ✓
```

### 2.5 Tokenização Qualitativa

| Palavra | Tokens | Fragmentação |
|---|---|---|
| `astronomia` | 2 | `▁as` + `tronomia` |
| `computação` | 1 | completa |
| `linguagem` | 1 | completa |
| `transformer` | 3 | `▁trans` + `for` + `mer` |
| `futebol` | 1 | completa |
| `Noruega` | 1 | completa |
| `Brasil` | 1 | completa |
| `português` | 1 | completa |
| `coração` | 1 | completa |
| `inteligência` | 1 | completa |
| `Python` | 2 | `▁Py` + `thon` |
| `redes neurais` | 4 | `▁redes` + `▁ne` + `ura` + `is` |

✓ Subwords naturais, sem fragmentação excessiva.

### 2.6 Pipeline GPT-like

```python
article = "O Brasil é um país localizado na América do Sul."
ids = sp.encode(article)                            # [83, 510, 72, ...]
full = ids + [sp.piece_to_id('<eos>')]              # + [2]
decoded = sp.decode(full)                           # "O Brasil é um país...<eos>"
```

✓ Funcional. O `<eos>` é corretamente inserido como id=2 e decodificado como `<eos>`.

---

## 3. Problemas

| # | Severidade | Descrição | Status |
|---|---|---|---|
| 1 | **Média** | `bos_id()`/`eos_id()`/`pad_id()` retornam `-1` (SentencePiece + `user_defined_symbols`). | Mitigado: `piece_to_id('<eos>')` = 2 |
| 2 | **Info** | `encode('<eos>')` adiciona prefixo `▁` (id 13501). | Sem impacto: DataLoader usa `piece_to_id()`, não `encode()` |

---

## 4. Conclusão

**PASS WITH WARNINGS** — Tokenizer pronto para tokenização do corpus e treino GPT-like.
