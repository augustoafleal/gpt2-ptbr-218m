#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

PYTHON="python3"
if [ -x "$ROOT/venv/bin/python" ]; then
    PYTHON="$ROOT/venv/bin/python"
fi

if [ $# -ne 1 ]; then
    echo "Usage: $0 <timestamp>"
    echo "Example: $0 20260529_212545"
    exit 1
fi

TIMESTAMP="$1"
CHECKPOINT="$ROOT/runs/$TIMESTAMP/best.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: checkpoint not found: $CHECKPOINT"
    exit 1
fi

OUTDIR="$ROOT/runs/$TIMESTAMP/inference"
mkdir -p "$OUTDIR"

FILES=(
    "astronomy.txt:Astronomia"
    "astronomy_extended.txt:A astronomia é uma ciência natural que estuda os corpos celestes e os fenômenos que ocorrem fora da atmosfera da Terra."
    "brazil.txt:Brasil é um país da América do Sul."
    "ai.txt:Inteligência artificial é um campo da ciência da computação."
    "history.txt:A Segunda Guerra Mundial foi"
    "football.txt:O futebol é um esporte coletivo."
    "earth.txt:A Terra é um planeta."
    "porto_alegre.txt:Porto Alegre é a capital."
)

GENERATED_FILES=()

for entry in "${FILES[@]}"; do
    filename="${entry%%:*}"
    prompt="${entry#*:}"

    echo "Generating $filename ..."

    output=$("$PYTHON" "$ROOT/scripts/generate_text.py" \
        --checkpoint "$CHECKPOINT" \
        --prompt "$prompt" \
        --max-new-tokens 200 \
        --temperature 0.8 \
        --top-k 40 \
        --device cpu 2>/dev/null)

    generated=$(echo "$output" | sed '1,/^$/d')

    {
        echo "===================================================="
        echo "PROMPT"
        echo "===================================================="
        echo
        echo "$prompt"
        echo
        echo "===================================================="
        echo "OUTPUT"
        echo "===================================================="
        echo
        echo "$generated"
    } > "$OUTDIR/$filename"

    GENERATED_FILES+=("$filename")
done

cat > "$OUTDIR/metadata.json" <<EOF
{
  "checkpoint": "runs/$TIMESTAMP/best.pt",
  "temperature": 0.8,
  "top_k": 40,
  "max_new_tokens": 200,
  "generated_files": [
$(for f in "${GENERATED_FILES[@]}"; do
    echo "    \"$f\","
done | sed '$s/,$//')
  ]
}
EOF

echo ""
echo "Done. Output in: $OUTDIR"
