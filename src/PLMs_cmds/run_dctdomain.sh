set -e
QUERY_FA_PATH=
TARGET_FA_PATH=
RUN_BATCH=

DB_KHITS=45531
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
echo "Using the DCT_DOMAIN_SRC_DIR: $DCT_DOMAIN_SRC_DIR"
echo "Using the number of workers: $DCT_DOMAIN_NUM_PARALLEL"
echo "Using the GPU_DEVICES: $DCT_DOMAIN_GPU_DEVICES"
NUM_GPUS=$(echo $DCT_DOMAIN_GPU_DEVICES | awk -F',' '{print NF}')
if [ -z "$BASE_DIR" ]; then
    echo "Error: Data paths are not set. Please set it by 'source PLMs_cmds/.env' before running the script."
fi

export PATH="$(dirname $DCT_DOMAIN_PYTHON_PATH):$PATH"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "=============init env end ==========="

query_basename=$(basename "${QUERY_FA_PATH%.fa}")
target_basename=$(basename "${TARGET_FA_PATH%.fa}")

OUTPUT_DIR="$TEMP_DIR/result_dctdomain_${query_basename}_to_${target_basename}"
TEMP_DATA_SPLIT_DIR="$TEMP_DIR/${query_basename}_batch"
mkdir -p "$OUTPUT_DIR"

echo "Keeping the number of hits for DCTdomain db search: $DB_KHITS"
echo "Keeping the number of hits for parser: $PARSER_KHITS"
if [ "$DB_KHITS" -lt "$PARSER_KHITS" ]; then
    echo "DB_KHITS ($DB_KHITS) must be greater than or equal to PARSER_KHITS ($PARSER_KHITS)" >&2
    exit 1
fi

set_dctdomain_target_db_path() {
    echo "============start generating target DB path ============"
    echo "generate the TARGET_DB_PATH from $TARGET_FA_PATH"
    
    DCT_DOMAIN_TARGET_DB_PATH="$DATA_DIR/db/db_${target_basename}_dctdomain/${target_basename}_db.db"
    mkdir -p $(dirname "$DCT_DOMAIN_TARGET_DB_PATH")

    if [ ! -f "$DCT_DOMAIN_TARGET_DB_PATH" ]; then
        echo "Creating directory for ${query_basename} database..."
        CUDA_VISIBLE_DEVICES=$DCT_DOMAIN_GPU_DEVICES python ${DCT_DOMAIN_SRC_DIR}/src/make_db.py \
        --fafile $TARGET_FA_PATH \
        --dbfile $DCT_DOMAIN_TARGET_DB_PATH \
        --gpu $NUM_GPUS \
        --cpu $DCT_DOMAIN_NUM_PARALLEL \
        || { echo "Creating DB failed"; exit 1; }
    else
        echo "Database for ${query_basename} already exists. Skipping creation."
    fi
}      

split_fasta_files() {
    echo "============start splitting============"
    echo "FA file path: $QUERY_FA_PATH"
    echo "query basename: $query_basename"
    if [ ! -d ${TEMP_DATA_SPLIT_DIR} ]; then
        echo "${TEMP_DATA_SPLIT_DIR} not found. Splitting the fasta file..."
        python $BASE_DIR/src/utils/split_fasta_batches.py --fasta_path $QUERY_FA_PATH --output_dir  $TEMP_DATA_SPLIT_DIR \
        || { echo "Splitting batch failed"; exit 1; } 
    else
        echo "FASTA already split into batches. Skipping this step."
    fi
}

run_dctdomain_search() {
    query_file_path="$1"
    output_file_path="$2"

    CUDA_VISIBLE_DEVICES=$DCT_DOMAIN_GPU_DEVICES \
    python ${DCT_DOMAIN_SRC_DIR}/src/query_db.py \
        --query $query_file_path \
        --db $DCT_DOMAIN_TARGET_DB_PATH \
        --out $output_file_path \
        --khits $DB_KHITS \
        --gpu $NUM_GPUS \
        --cpu 1 \
        || { echo "Query failed"; exit 1; }
    wait
}

run_batch() {
    split_fasta_files
    N=0
    BATCHES=("$TEMP_DATA_SPLIT_DIR"/*.fa)

    for batch in "${BATCHES[@]}"; do
        {
            echo "Processing batch: $batch"
            run_dctdomain_search "$batch" "$OUTPUT_DIR/$(basename $batch .fa).txt"
        } &

        let N=N+1
        if [[ $N -ge $DCT_DOMAIN_NUM_PARALLEL ]]; then
            wait
            N=0
        fi
    done
    wait
}

set_dctdomain_target_db_path

echo "============start dctdomain batch query============"
if [ -z "$RUN_BATCH" ]; then
    echo "Processing single file: $QUERY_FA_PATH"
    run_dctdomain_search "$QUERY_FA_PATH" "$OUTPUT_DIR/${query_basename}.txt"
else
    echo "Running Batch dctdomain for $query_basename with database $DCT_DOMAIN_TARGET_DB_PATH"
    run_batch
fi


echo "============parse query target scores from original output files============"
export PATH="$(dirname $EVALPLMS_PYTHON_PATH):$PATH"
python $BASE_DIR/src/utils/parser.py --data_type dctdomain --input_path $OUTPUT_DIR --save_path $DATA_DIR --max_hits $PARSER_KHITS\
|| { echo "Parsing info failed"; exit 1; }

echo "============dctdomain query completed============"

