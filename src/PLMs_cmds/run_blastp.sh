set -e
QUERY_FA_PATH=
TARGET_FA_PATH=
DB_KHITS=15177
PARSER_KHITS=15177
EVALUE=10000000000


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
echo "Using the number of workers: $BLASTP_NUM_PARALLEL"
if [ -z "$BASE_DIR" ]; then
    echo "Error: Data paths are not set. Please set it by 'source PLMs_cmds/.env' before running the script."
fi

export PATH="$(dirname $BLASTP_PYTHON_PATH):$PATH"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "=============init env end ==========="

query_basename="$(basename "$QUERY_FA_PATH" .fa)"
target_basename="$(basename "$TARGET_FA_PATH" .fa)"

TEMP_DATA_SPLIT_DIR="$TEMP_DIR/${query_basename}_batch"
OUTPUT_DIR="$TEMP_DIR/result_blastp_${query_basename}_to_${target_basename}_blosum62_Q11R1"
echo "Keeping the number of hits for BLASTP db search: $DB_KHITS"
echo "Keeping the number of hits for parser: $PARSER_KHITS"
if [ "$DB_KHITS" -lt "$PARSER_KHITS" ]; then
    echo "Assertion failed: DB_KHITS ($DB_KHITS) must be greater than or equal to PARSER_KHITS ($PARSER_KHITS)" >&2
    exit 1
fi

set_blastp_target_db_path() {
    echo "============start generating target DB path ============"
    echo "generate the TARGET_DB_PATH from $TARGET_FA_PATH"
    target_basename=$(basename "${TARGET_FA_PATH%.fa}")
    BLASTP_TARGET_DB_DIR="$BASE_DIR/data/db/db_${target_basename}_blastp"
    BLASTP_TARGET_DB_PATH="${BLASTP_TARGET_DB_DIR}/${target_basename}_db"
    if [ ! -d "$BLASTP_TARGET_DB_DIR" ]; then
        echo "Creating directory for ${query_basename} database..."
        mkdir -p "$BLASTP_TARGET_DB_DIR"
        makeblastdb -in $TARGET_FA_PATH -parse_seqids -dbtype prot -out "$BLASTP_TARGET_DB_PATH" \
        || { echo "makeblastdb failed"; exit 1; }
    else
        echo "Database for ${query_basename} already exists. Skipping creation."
    fi
}


set_blastp_target_db_path

echo "============start splitting============"
echo "FA file path: $QUERY_FA_PATH"
echo "query basename: $query_basename"
if [ ! -d ${TEMP_DATA_SPLIT_DIR} ]; then
    echo "${TEMP_DATA_SPLIT_DIR} not found. Splitting the fasta file..."
    python $BASE_DIR/src/utils/split_fasta_batches.py --fasta_path $QUERY_FA_PATH --output_dir  $TEMP_DATA_SPLIT_DIR \
    || { echo "Batch split failed"; exit 1; }
else
    echo "FASTA already split into batches. Skipping this step."
fi

echo "============start blastp batch query============"

N=0
BATCHES=("$TEMP_DATA_SPLIT_DIR"/*.fa)
mkdir -p "$OUTPUT_DIR"

echo "Running blastp for $query_basename with database $BLASTP_TARGET_DB_PATH"
for batch in "${BATCHES[@]}"; do
    {
        echo "Processing batch: $batch"
        blastp \
            -num_alignments 0 \
            -num_descriptions $DB_KHITS \
            -gapopen 11 \
            -gapextend 1 \
            -evalue $EVALUE \
            -matrix BLOSUM62 \
            -comp_based_stats -2 \
            -db $BLASTP_TARGET_DB_PATH \
            -query "$batch" \
            -max_hsps 1 \
            > "$OUTPUT_DIR/$(basename "${batch%.fa}").out" || { echo "Blastp failed: ${batch}"; exit 1; }
    } &

    let N=N+1
    if [[ $N -ge $BLASTP_NUM_PARALLEL ]]; then
        wait
        N=0
    fi
done
wait


echo "============parse query target scores from original output files============"
python $BASE_DIR/src/utils/parser.py --data_type blastp --input_path $OUTPUT_DIR --save_path $DATA_DIR --max_hits $PARSER_KHITS\
|| { echo "Parsing info failed"; exit 1; }
echo "============blastp query completed============"