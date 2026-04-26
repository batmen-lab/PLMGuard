set -e
QUERY_FA_PATH=
TARGET_FA_PATH=
RUN_BATCH=
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
            echo "Usage: bash $0 -q <query_fasta_path> -t <target_fasta_path> [-k <db_khits>] [-p <parser_khits>]"
            exit 1
            ;;
    esac
done

if [[ -z "$QUERY_FA_PATH" ]]; then
    echo "Error: Missing required argument -q <query_fasta_path>"
    echo "Usage: bash $0 -q <query_fasta_path> [-t <target_fasta_path>] [-k <db_khits>] [-p <parser_khits>]"
    exit 1
fi

echo "============init env start============"
echo "Using the BASE_DIR: $BASE_DIR"
echo "Using the DATA_DIR: $DATA_DIR"
echo "Using the TEMP_DIR: $TEMP_DIR"
echo "Using the PLMSEARCH_SRC_DIR: $PLMSEARCH_SRC_DIR"
echo "Using the number of workers: $PLMSEARCH_NUM_PARALLEL"
echo "Using the GPU_DEVICES: $PLMSEARCH_GPU_DEVICES"
export PATH="$(dirname $PLMSEARCH_PYTHON_PATH):$PATH"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "=============init env end ==========="


query_basename="$(basename "$QUERY_FA_PATH" .fa)"
target_basename="$(basename "$TARGET_FA_PATH" .fa)"

TEMP_DATA_SPLIT_DIR="$TEMP_DIR/${query_basename}_batch"
OUTPUT_DIR="$TEMP_DIR/result_plm_${query_basename}_to_${target_basename}"
mkdir -p "$OUTPUT_DIR"

set_target_db_path() {
    echo "============start setting target db path============"
    if [ -z "$TARGET_FA_PATH" ]; then
        echo "Using the default PLMSEARCH_TARGET_DB_PATH: $PLMSEARCH_TARGET_DB_PATH"
    else
        echo "============start generating target DB path ============"
        echo "generate the TARGET_DB_PATH from $TARGET_FA_PATH"

        PLMSEARCH_TARGET_DB_PATH="$BASE_DIR/data/db/db_${target_basename}_plm/${target_basename}_embedding.pkl"
        mkdir -p "$(dirname "$PLMSEARCH_TARGET_DB_PATH")"

        if [ ! -f "$PLMSEARCH_TARGET_DB_PATH" ]; then
            echo "Creating directory for ${target_basename} database..."
           CUDA_VISIBLE_DEVICES=$PLMSEARCH_GPU_DEVICES python "$PLMSEARCH_SRC_DIR/plmsearch/embedding_generate.py" \
                -emp "$PLMSEARCH_SRC_DIR/plmsearch_data/model/esm/esm1b_t33_650M_UR50S.pt" \
                -f "$TARGET_FA_PATH" \
                -e "$PLMSEARCH_TARGET_DB_PATH" \
                || { echo "Creating db failed"; exit 1; }
        else
            echo "Database for ${target_basename} already exists. Skipping creation."
        fi
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

run_plmsearch() {
    query_file_path="$1"
    output_file_path="$2"

    embed_name="$(basename "$query_file_path" .fa)"
    query_embed="$BASE_DIR/data/db/db_astral_plm/db_${embed_name}_embedding.pkl"

    echo "Embedding will be saved to: $query_embed"
    CUDA_VISIBLE_DEVICES=$PLMSEARCH_GPU_DEVICES python "$PLMSEARCH_SRC_DIR/plmsearch/embedding_generate.py" \
        -emp "$PLMSEARCH_SRC_DIR/plmsearch_data/model/esm/esm1b_t33_650M_UR50S.pt" \
        -f "$query_file_path" \
        -e "$query_embed" \
        || { echo "Embedding generation failed"; exit 1; }

    # first_gpu=$(echo $PLMSEARCH_GPU_DEVICES | cut -d',' -f1)
    CUDA_VISIBLE_DEVICES=$PLMSEARCH_GPU_DEVICES python "$PLMSEARCH_SRC_DIR/plmsearch/main_similarity.py" \
        -iqe "$query_embed" \
        -ite "$PLMSEARCH_TARGET_DB_PATH" \
        -smp "$PLMSEARCH_SRC_DIR/plmsearch_data/model/plmsearch.sav" \
        -osr "$output_file_path" \
        -k "$DB_KHITS" \
        || { echo "Query failed"; exit 1; }
}

run_batch() {
    split_fasta_files
    N=0
    BATCHES=("$TEMP_DATA_SPLIT_DIR"/*.fa)
    for batch in "${BATCHES[@]}"; do
        {
            echo "Processing batch: $batch"
            run_plm_search "$batch" "$OUTPUT_DIR/$(basename $batch .fa).txt"
        } &

        let N=N+1
        if [[ $N -ge $PLM_NUM_PARALLEL ]]; then
            wait
            N=0
        fi
    done
    wait
}


set_target_db_path

echo "============start PLMSearch query============"
if [ -z "$RUN_BATCH" ]; then
    echo "Processing single file: $QUERY_FA_PATH"
    run_plmsearch "$QUERY_FA_PATH" "$OUTPUT_DIR/${query_basename}.txt"
else
    echo "Running Batch PLMSearch for $query_basename with database $PLMSEARCH_TARGET_DB_PATH"
    run_batch
fi

echo "============parse query target scores from original output files============"
export PATH="$(dirname $EVALPLMS_PYTHON_PATH):$PATH"

python $BASE_DIR/src/utils/parser.py \
    --data_type plm \
    --input_path $OUTPUT_DIR \
    --save_path $DATA_DIR \
    --max_hits $PARSER_KHITS \
    || { echo "Parsing info failed"; exit 1; }
echo "============PLMsearch query completed============"