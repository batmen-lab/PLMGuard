
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
echo "Using the TMVEC_SRC_DIR: $TMVEC_SRC_DIR"
echo "Using the number of workers: $TMVEC_NUM_PARALLEL"
echo "Using the GPU_DEVICES: $TMVEC_GPU_DEVICES"
NUM_GPUS=$(echo $TMVEC_GPU_DEVICES | awk -F',' '{print NF}')


export PATH="$(dirname $TMVEC_PYTHON_PATH):$PATH"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "=============init env end ==========="

query_basename="$(basename "$QUERY_FA_PATH" .fa)"
target_basename="$(basename "$TARGET_FA_PATH" .fa)"

TEMP_DATA_SPLIT_DIR="$TEMP_DIR/${query_basename}_batch"
OUTPUT_DIR="$TEMP_DIR/result_tmvec_${query_basename}_to_${target_basename}"
mkdir -p "$OUTPUT_DIR"

set_target_db_path() {
    echo "============start setting target db path============"
    echo "generate the TARGET_DB_PATH from $TARGET_FA_PATH"

    target_query_basename="$(basename "$TARGET_FA_PATH" .fa)"
    TMVEC_TARGET_DB_PATH="$BASE_DIR/data/db/db_${target_query_basename}_tmvec"
    # cp "$TARGET_FA_PATH" "$BASE_DIR/data/db/"
    # TMVEC_TARGET_FA_PATH="$BASE_DIR/data/db/$(basename "$TARGET_FA_PATH")"
    # mkdir -p "$TMVEC_TARGET_DB_PATH"

    if [ ! -d "$TMVEC_TARGET_DB_PATH" ]; then
        echo "Creating directory for ${query_basename} database..."
        CUDA_VISIBLE_DEVICES=$TMVEC_GPU_DEVICES python "${TMVEC_SRC_DIR}/scripts/tmvec-build-database" \
            --input-fasta "$TARGET_FA_PATH" \
            --tm-vec-model "${TMVEC_SRC_DIR}/model/tm_vec_cath_model.ckpt" \
            --tm-vec-config-path "${TMVEC_SRC_DIR}/model/tm_vec_cath_model_params.json" \
            --protrans-model "Rostlab/prot_t5_xl_half_uniref50-enc" \
            --device "gpu" \
            --output "$TMVEC_TARGET_DB_PATH" \
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

run_tmvec_search() {
    query_file_path="$1"
    output_file_path="$2"

    CUDA_VISIBLE_DEVICES=$TMVEC_GPU_DEVICES python "${TMVEC_SRC_DIR}/scripts/tmvec-search" \
            --query "$query_file_path" \
            --output "$output_file_path" \
            --tm-vec-model "${TMVEC_SRC_DIR}/model/tm_vec_cath_model.ckpt" \
            --tm-vec-config "${TMVEC_SRC_DIR}/model/tm_vec_cath_model_params.json" \
            --database "$TMVEC_TARGET_DB_PATH/db.npy" \
            --metadata "$TMVEC_TARGET_DB_PATH/meta.npy" \
            --database-fasta "$TARGET_FA_PATH" \
            --protrans-model "Rostlab/prot_t5_xl_half_uniref50-enc" \
            --device "gpu" \
            --output-format "tabular" \
            --k-nearest-neighbors $DB_KHITS \
            || { echo "Query failed"; exit 1; }
}

run_batch() {
    split_fasta_files

    N=0
    BATCHES=("$TEMP_DATA_SPLIT_DIR"/*.fa)
    mkdir -p "$OUTPUT_DIR"
    for batch in "${BATCHES[@]}"; do
        {
            echo "Processing batch: $batch"
            output_path="$OUTPUT_DIR/$(basename $batch .fa).txt"
            run_tmvec_search "$batch" "$output_path"
        } &

        let N=N+1
        if [[ $N -ge $TMVEC_NUM_PARALLEL ]]; then
            wait
            N=0
        fi
    done
    wait
}

set_target_db_path

echo "============start tmvec query============"
if [ -z "$RUN_BATCH" ]; then
    echo "Processing single file: $QUERY_FA_PATH"
    run_tmvec_search "$QUERY_FA_PATH" "$OUTPUT_DIR/${query_basename}.txt"
else
    echo "Running Batch tmvec for $query_basename with database $TMVEC_TARGET_DB_PATH"
    run_batch
fi

echo "============parse query target scores from original output files============"
export PATH="$(dirname $EVALPLMS_PYTHON_PATH):$PATH"
python $BASE_DIR/src/utils/parser.py \
    --data_type tmvec \
    --input_path $OUTPUT_DIR \
    --save_path $DATA_DIR \
    --max_hits $PARSER_KHITS \
    || { echo "Parsing info failed"; exit 1; }
echo "============tmvec query completed============"




