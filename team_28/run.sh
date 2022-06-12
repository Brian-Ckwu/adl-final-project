eval "$(conda shell.bash hook)"

echo "------- Creating conda environment and installing package... -------"
conda env create -f environment.yml
echo "------- Done! -------"

echo "------- Activating environment -------"
conda activate team_28
echo "------- Done! -------"

echo "------- Running simulator and generating output dialogue... -------"
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.3.0/en_core_web_sm-3.3.0.tar.gz
python simulator.py \
    --output ./output.jsonl \
    --disable_output_dialog \
    --num_generation 20