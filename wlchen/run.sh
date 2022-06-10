export CUDA_VISIBLE_DEVICES=0

echo "------- Creating conda environment and installing package... -------"
conda env create -f environment.yml
echo "------- Done! -------"

echo "------- Activating environment -------"
eval "$(conda shell.bash hook)"
conda activate team_28
echo "------- Done! -------"

echo "------- Running simulator and generating output dialogue... -------"
python -m spacy download en_core_web_sm
python simulator.py \
    --model_name_or_path facebook/blenderbot-400M-distill \
    --output output.jsonl \
    --disable_output_dialog
echo "------- Done! -------"