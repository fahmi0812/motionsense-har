#!/usr/bin/env bash
# Jalankan seluruh training. Pakai:  bash run_training.sh          (CSV di data/privat_csv)
#                                    bash run_training.sh --dummy  (buat data DUMMY dulu, hanya untuk uji)
#                                    bash run_training.sh --supabase
set -e
cd "$(dirname "$0")/training"
python 01_siapkan_publik.py
if [ "$1" == "--dummy" ]; then python make_dummy_private.py; python 02_siapkan_privat.py
elif [ "$1" == "--supabase" ]; then python 02_siapkan_privat.py --supabase
else python 02_siapkan_privat.py --csv ../data/privat_csv; fi
python 03_latih.py
python 04_ekspor_onnx.py
echo "SELESAI. Jalankan web: bash run_web.sh"
