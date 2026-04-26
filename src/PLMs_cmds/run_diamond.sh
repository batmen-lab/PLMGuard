set -e
QUERY_FA_PATH=
TARGET_FA_PATH=
EVALUE=10000000000
DB_KHITS=15177
PARSER_KHITS=15177

while getopts "q:t:k:p:" opt; do
    case $opt in
        q)
            QUERY_FA_PATH="$OPTARG"
            ;;
        t)
            TARGET_FA_PATH="$OPTARG"
            ;;
        k)
            DB_KHITS="$OPTARG"
            ;;
        p)
            PARSER_KHITS="$OPTARG"
            ;;
        *)
            echo "Usage: bash $0 -q <fasta_file_name> -t <query_target_fasta_path> [-k <db_khits>] [-p <parser_khits>]"
            exit 1
            ;;
    esac
done

if [[ -z "$QUERY_FA_PATH" ]]; then
    echo "Error: Missing required argument -q <fasta_file_name>"
    echo "Usage: bash $0 -q <fasta_file_name> [-t <query_target_fasta_path>] [-k <db_khits>] [-p <parser_khits>]"
    exit 1
fi

echo "============init env start============"
echo "Using the BASE_DIR: $BASE_DIR"
echo "Using the DATA_DIR: $DATA_DIR"
echo "Using the TEMP_DIR: $TEMP_DIR"
echo "Using the number of workers: $DIAMOND_NUM_PARALLEL"
if [ -z "$BASE_DIR" ]; then
    echo "Error: Data paths are not set. Please set it by 'source PLMs_cmds/.env' before running the script."
fi

export PATH="$(dirname $DIAMOND_PYTHON_PATH):$PATH"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "=============init env end ==========="

query_basename="$(basename "$QUERY_FA_PATH" .fa)"
target_basename="$(basename "$TARGET_FA_PATH" .fa)"

TEMP_DATA_SPLIT_DIR="$TEMP_DIR/${query_basename}_batch"
OUTPUT_DIR="$TEMP_DIR/result_diamond_${query_basename}_to_${target_basename}_blosum62_Q11R1"
mkdir -p "$OUTPUT_DIR"
echo "Keeping the number of hits for BLASTP db search: $DB_KHITS"
echo "Keeping the number of hits for parser: $PARSER_KHITS"
if [ "$DB_KHITS" -lt "$PARSER_KHITS" ]; then
    echo "Assertion failed: DB_KHITS ($DB_KHITS) must be greater than or equal to PARSER_KHITS ($PARSER_KHITS)" >&2
    exit 1
fi

set_diamond_target_db_path() {
    echo "============start generating target DB path ============"
    echo "generate the TARGET_DB_PATH from $TARGET_FA_PATH"
    target_basename=$(basename "${TARGET_FA_PATH%.fa}")
    DIAMOND_TARGET_DB_DIR="$BASE_DIR/data/db/db_${target_basename}_diamond"
    DIAMOND_TARGET_DB_PATH="${DIAMOND_TARGET_DB_DIR}/${target_basename}_db"
    if [ ! -d "$DIAMOND_TARGET_DB_DIR" ]; then
        echo "Creating directory for ${query_basename} database..."
        mkdir -p "$DIAMOND_TARGET_DB_DIR"
        "${DIAMOND_SRC_DIR}/diamond" "makedb" \
        --in $TARGET_FA_PATH \
        -d "$DIAMOND_TARGET_DB_PATH" || { echo "makeblastdb failed"; exit 1; }
    else
        echo "Database for ${query_basename} already exists. Skipping creation."
    fi
}

set_diamond_target_db_path

echo "============start diamond batch query============"

"${DIAMOND_SRC_DIR}/diamond" "blastp" \
    -d "$DIAMOND_TARGET_DB_PATH" \
    -q "$QUERY_FA_PATH" \
    -o "$OUTPUT_DIR/result.tsv" \
    --gapopen 11 \
    --gapextend 1 \
    --max-target-seqs $DB_KHITS \
    --evalue $EVALUE \
    --matrix BLOSUM62 \
    --ultra-sensitive \
    --unal 0 \
    --outfmt 6 qseqid sseqid score bitscore evalue || { echo "Diamond failed: ${batch}"; exit 1; }
echo "============finished diamond batch query============"

echo "============parse query target scores from original output files============"
python $BASE_DIR/src/utils/parser.py --data_type diamond --input_path $OUTPUT_DIR --save_path $DATA_DIR --max_hits $PARSER_KHITS\
|| { echo "Parsing info failed"; exit 1; }
echo "============diamond query completed============"