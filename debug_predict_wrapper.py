import json
import shutil
import sys

from allennlp.commands import main

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "predict",
    "storage/model.tar.gz",
    "bad_review.json",
    "--include-package", "allennlp_imdb",
    "--predictor", "imdb",
]

main()