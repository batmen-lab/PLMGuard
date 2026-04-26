set -e
QUERY_FA_PATH=
TARGET_FA_PATH=

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
echo "Using the number of workers: $NEAR_NUM_PARALLEL"
if [ -z "$BASE_DIR" ]; then
    echo "Error: Data paths are not set. Please set it by 'source PLMs_cmds/.env' before running the script."
fi

export PATH="$(dirname $NEAR_PYTHON_PATH):$PATH"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "=============init env end ==========="

query_basename="$(basename "$QUERY_FA_PATH" .fa)"
target_basename="$(basename "$TARGET_FA_PATH" .fa)"

TEMP_DATA_SPLIT_DIR="$TEMP_DIR/${query_basename}_batch"
OUTPUT_DIR="$TEMP_DIR/result_near_${query_basename}_to_${target_basename}"
echo "Keeping the number of hits for parser: $PARSER_KHITS"
if [ "$DB_KHITS" -lt "$PARSER_KHITS" ]; then
    echo "Assertion failed: DB_KHITS ($DB_KHITS) must be greater than or equal to PARSER_KHITS ($PARSER_KHITS)" >&2
    exit 1
fi

set_near_target_db_path() {
    echo "============start generating target DB path ============"
    echo "generate the TARGET_DB_PATH from $TARGET_FA_PATH"
    NEAR_TARGET_DB_DIR="$BASE_DIR/data/db/db_${target_basename}_near"
    NEAR_TARGET_DB_PATH="${NEAR_TARGET_DB_DIR}/${target_basename}_db.npz"
    if [ ! -d "$NEAR_TARGET_DB_DIR" ]; then
        echo "Creating directory for ${target_basename} database..."
        mkdir -p "$NEAR_TARGET_DB_DIR"
        python "$NEAR_SRC_DIR/src/near_embed.py" \
        "$NEAR_SRC_DIR/resnet.yaml" \
        "$NEAR_SRC_DIR/models/resnet_877_256.pt" \
        $TARGET_FA_PATH $NEAR_TARGET_DB_PATH \
        || { echo "makedb failed"; exit 1; }
    else
        echo "Database for ${target_basename} already exists. Skipping creation."
    fi
}

set_near_qry_db_path() {
    echo "============start generating query DB path ============"
    echo "generate the QUERY_DB_PATH from $QUERY_FA_PATH"
    NEAR_QUERY_DB_DIR="$BASE_DIR/data/db/db_${query_basename}_near"
    NEAR_QUERY_DB_PATH="${NEAR_QUERY_DB_DIR}/${query_basename}_db.npz"
    if [ ! -d "$NEAR_QUERY_DB_DIR" ]; then
        echo "Creating directory for ${query_basename} database..."
        mkdir -p "$NEAR_QUERY_DB_DIR"
        python "$NEAR_SRC_DIR/src/near_embed.py" \
        "$NEAR_SRC_DIR/resnet.yaml" \
        "$NEAR_SRC_DIR/models/resnet_877_256.pt" \
        $QUERY_FA_PATH $NEAR_QUERY_DB_PATH \
        || { echo "makedb failed"; exit 1; }
    else
        echo "Database for ${target_basename} already exists. Skipping creation."
    fi
}


set_near_target_db_path
set_near_qry_db_path


echo "============start near query============"

if [ ! -f "$OUTPUT_DIR/hits.csv" ] && [ ! -f "$OUTPUT_DIR/sorted_hits.csv" ]; then
    echo "Running near for $query_basename with database $NEAR_TARGET_DB_PATH"
    mkdir -p "$OUTPUT_DIR"  
    python "$NEAR_SRC_DIR/src/search.py" \
        -g \
        -q $NEAR_QUERY_DB_PATH \
        -t $NEAR_TARGET_DB_PATH \
        -o "$OUTPUT_DIR/hits.csv" \

else
    echo "NEAR output directory ${OUTPUT_DIR}/hits.csv already exists. Skipping NEAR search."
fi   


echo "============parse query target scores from original output files============"
if [ ! -f "$OUTPUT_DIR/sorted_hits.csv" ]; then
    echo "Converting NEAR output to sorted_hits.csv"
    python $BASE_DIR/src/utils/convert_near.py \
        --input_path "$OUTPUT_DIR/hits.csv" \
        --query_fasta $QUERY_FA_PATH \
        --target_fasta $TARGET_FA_PATH \
        --max_hits $DB_KHITS || { echo "Converting near output failed"; exit 1; }

    rm "$OUTPUT_DIR/hits.csv"
fi

python $BASE_DIR/src/utils/parser.py \
    --data_type near \
    --input_path $OUTPUT_DIR \
    --save_path $DATA_DIR \
    --max_hits $PARSER_KHITS || { echo "Parsing info failed"; exit 1; }

echo "============near query completed============"