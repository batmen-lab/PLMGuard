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
echo "Using the number of workers: $MMSEQ2_NUM_PARALLEL"
if [ -z "$BASE_DIR" ]; then
    echo "Error: Data paths are not set. Please set it by 'source PLMs_cmds/.env' before running the script."
fi

export PATH="$(dirname $MMSEQ2_PYTHON_PATH):$PATH"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "=============init env end ==========="

query_basename="$(basename "$QUERY_FA_PATH" .fa)"
target_basename="$(basename "$TARGET_FA_PATH" .fa)"

TEMP_DATA_SPLIT_DIR="$TEMP_DIR/${query_basename}_batch"
OUTPUT_DIR="$TEMP_DIR/result_mmseq2_${query_basename}_to_${target_basename}"
echo "Keeping the number of hits for mmseq2 db search: $DB_KHITS"
echo "Keeping the number of hits for parser: $PARSER_KHITS"
if [ "$DB_KHITS" -lt "$PARSER_KHITS" ]; then
    echo "Assertion failed: DB_KHITS ($DB_KHITS) must be greater than or equal to PARSER_KHITS ($PARSER_KHITS)" >&2
    exit 1
fi

set_mmseq2_target_db_path() {
    echo "============start generating target DB path ============"
    echo "generate the TARGET_DB_PATH from $TARGET_FA_PATH"
    MMSEQ2_TARGET_DB_DIR="$BASE_DIR/data/db/db_${target_basename}_mmseq2"
    MMSEQ2_TARGET_DB_PATH="${MMSEQ2_TARGET_DB_DIR}/${target_basename}_db"
    if [ ! -d "$MMSEQ2_TARGET_DB_DIR" ]; then
        echo "Creating directory for ${target_basename} database..."
        mkdir -p "$MMSEQ2_TARGET_DB_DIR"
        mmseqs "createdb" $TARGET_FA_PATH $MMSEQ2_TARGET_DB_PATH \
        || { echo "makedb failed"; exit 1; }
    else
        echo "Database for ${target_basename} already exists. Skipping creation."
    fi
}

set_mmseq2_qry_db_path() {
    echo "============start generating query DB path ============"
    echo "generate the QUERY_DB_PATH from $QUERY_FA_PATH"
    MMSEQ2_QUERY_DB_DIR="$BASE_DIR/data/db/db_${query_basename}_mmseq2"
    MMSEQ2_QUERY_DB_PATH="${MMSEQ2_QUERY_DB_DIR}/${query_basename}_db"
    if [ ! -d "$MMSEQ2_QUERY_DB_DIR" ]; then
        echo "Creating directory for ${query_basename} database..."
        mkdir -p "$MMSEQ2_QUERY_DB_DIR"
        mmseqs "createdb" $QUERY_FA_PATH $MMSEQ2_QUERY_DB_PATH \
        || { echo "makedb failed"; exit 1; }
    else
        echo "Database for ${target_basename} already exists. Skipping creation."
    fi
}


set_mmseq2_target_db_path
set_mmseq2_qry_db_path


echo "============start mmseq2 batch query============"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Running mmseq2 for $query_basename with database $MMSEQ2_TARGET_DB_PATH"
    TMP_OUTPUT_DIR="$OUTPUT_DIR/tmp/$query_basename"
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$TMP_OUTPUT_DIR/tmp"    
    mmseqs "search" $MMSEQ2_QUERY_DB_PATH $MMSEQ2_TARGET_DB_PATH $TMP_OUTPUT_DIR "$TMP_OUTPUT_DIR/tmp/${query_basename}_tmp" \
        --e-profile $EVALUE \
         -e $EVALUE \
        --max-seqs $DB_KHITS \
        --min-ungapped-score 0 \
        -s 7.5 || { echo "mmseq2 failed: ${batch}"; exit 1; }

    mmseqs "convertalis" $MMSEQ2_QUERY_DB_PATH $MMSEQ2_TARGET_DB_PATH $TMP_OUTPUT_DIR \
        "$OUTPUT_DIR/astral_result.tsv" \
        --format-output "query,target,evalue,bits"

    rm -rf "$OUTPUT_DIR/tmp" 
else
    echo "MMseq2 output directory $OUTPUT_DIR already exists. Skipping mmseq2 search."
fi   


echo "============parse query target scores from original output files============"
python $BASE_DIR/src/utils/parser.py --data_type mmseq2 --input_path $OUTPUT_DIR --save_path $DATA_DIR --max_hits $PARSER_KHITS\
|| { echo "Parsing info failed"; exit 1; }
echo "============mmseq2 query completed============"