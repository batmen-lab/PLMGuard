
QUERY_TSV_PATH=
TARGET_TSV_PATH=
DB_KHITS=15177
PARSER_KHITS=15177

while getopts "q:t:k:p:" opt; do
    case $opt in
        q)
            QUERY_TSV_PATH="$OPTARG"
            ;;
        t)
            TARGET_TSV_PATH="$OPTARG"
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

if [[ -z "$QUERY_TSV_PATH" ]]; then
    echo "Error: Missing required argument -q <fasta_file_name>"
    echo "Usage: bash $0 -q <fasta_file_name> [-t <query_target_fasta_path>] [-k <db_khits>] [-p <parser_khits>]"
    exit 1
fi

echo "============init env start============"
echo "Using the BASE_DIR: $BASE_DIR"
echo "Using the DATA_DIR: $DATA_DIR"
echo "Using the TEMP_DIR: $TEMP_DIR"
echo "Using the DHR_SRC_DIR: $DHR_SRC_DIR"
echo "Using the number of workers: $DHR_NUM_PARALLEL"
echo "Using the GPU_DEVICES: $DHR_GPU_DEVICES"
NUM_GPUS=$(echo $DHR_GPU_DEVICES | awk -F',' '{print NF}')
if [ -z "$BASE_DIR" ]; then
    echo "Error: Data paths are not set. Please set it by 'source PLMs_cmds/.env' before running the script."
fi

CKPT_PATH="$DHR_SRC_DIR/dhr2_ckpt"

export PATH="$(dirname $DHR_PYTHON_PATH):$PATH"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "=============init env end ==========="


query_basename="$(basename "$QUERY_TSV_PATH" .tsv)"
target_basename="$(basename "$TARGET_TSV_PATH" .tsv)"

TEMP_DATA_SPLIT_DIR="$TEMP_DIR/${query_basename}_batch"
OUTPUT_DIR="$TEMP_DIR/result_dhr_${query_basename}_to_${target_basename}"


set_target_db_path() {
    echo "============start setting target db path============"
    if [ -z "$TARGET_TSV_PATH" ]; then
        echo "Using the default DHR_TARGET_DB_DIR_PATH: $DHR_TARGET_DB_DIR_PATH"
    else
        echo "============start generating target DB path ============"
        echo "generate the TARGET_DB_PATH from $TARGET_TSV_PATH"
        DHR_TARGET_DB_DIR_PATH="$DATA_DIR/db/db_${target_basename}_dhr"
        
        if [ ! -d "$DHR_TARGET_DB_DIR_PATH/agg" ]; then
            echo "Creating directory for ${target_basename} database..."
            ls $TARGET_TSV_PATH
            
            CUDA_VISIBLE_DEVICES=$DHR_GPU_DEVICES python ${DHR_SRC_DIR}/do_embedding.py trainer.ur90_path=$(realpath "$TARGET_TSV_PATH") model.ckpt_path=$CKPT_PATH hydra.run.dir=$DHR_TARGET_DB_DIR_PATH \
            || { echo "Setting db failed: do_embedding.py"; exit 1; }

            CUDA_VISIBLE_DEVICES=$DHR_GPU_DEVICES python ${DHR_SRC_DIR}/do_agg.py -s ${TARGET_TSV_PATH} -e ${DHR_TARGET_DB_DIR_PATH}/ebd -o ${DHR_TARGET_DB_DIR_PATH}/agg \
            || { echo "Setting db failed: do_agg.py"; exit 1; }
        else
            echo "Database for ${target_basename} already exists. Skipping creation."
        fi
    fi
}

set_target_db_path

echo "============start DHR query============"
mkdir -p "$OUTPUT_DIR"
echo "Running DHR for $query_basename with database $DHR_TARGET_DB_DIR_PATH"
OUTPUT_PATH="$OUTPUT_DIR/${query_basename}.txt"

CUDA_VISIBLE_DEVICES=$DHR_GPU_DEVICES \
    python ${DHR_SRC_DIR}/do_retrieval.py \
    -i "$QUERY_TSV_PATH" \
    -d "$DHR_TARGET_DB_DIR_PATH/agg" \
    -o "$OUTPUT_PATH" \
    -n "$DB_KHITS" \
    || { echo "Query failed"; exit 1; }


echo "============parse query target scores from original output files============"
export PATH="$(dirname $EVALPLMS_PYTHON_PATH):$PATH"
python $BASE_DIR/src/utils/parser.py --data_type dhr --input_path $OUTPUT_DIR --save_path $DATA_DIR --max_hits $PARSER_KHITS\
|| { echo "Parsing info failed"; exit 1; }
echo "============dhr query completed============"

