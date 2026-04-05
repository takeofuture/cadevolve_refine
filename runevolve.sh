#rm code_db.json
#rm -rf embeddings
#rm -rf tmp_render
#rm -rf *_cases
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate cadevolve_occ
source envset.sh
xvfb-run -a python envtest.py
xvfb-run -a python pipeline.py 2>&1 | tee pipeline.log
