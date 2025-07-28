#!/bin/bash
set -euo pipefail
shopt -s nullglob

# SRC_DIR=$(realpath "../../examples")
# DEST_DIR=$(realpath ".")

SRC_DIR="../../../examples"
DEST_DIR="."

EXAMPLES=(im_cloudmodel im_ats)

for ex in "${EXAMPLES[@]}"; do
    dest="$DEST_DIR/$ex"
    src="$SRC_DIR/$ex"
    mkdir -p "$dest"
    cd "$dest"

    for notebook in "$src"/kim*.ipynb "$src"/postprocess*.ipynb; do
        # Globs that match nothing are removed by nullglob
        ln -sfn "$notebook" "$(basename "$notebook")"
        echo "Symlinked: $notebook -> /$(basename "$notebook")"
        # ln -sfn "$notebook" "$dest/$(basename "$notebook")"
        # echo "Symlinked: $notebook -> $dest/$(basename "$notebook")"
    done

    cd ../
done

# # Define source and destination directories
# SRC_DIR="../../examples"
# DEST_DIR="."

# EXAMPLES=("im_cloudmodel" "im_ats" "fm_evapotranspiration")

# # Ensure the destination directories exists
# for ex in "${EXAMPLES[@]}"; do
#     mkdir -p "$DEST_DIR/$(basename "$ex")"
# done

# # Find and symlink all ex*.ipynb files
# for ex in "${EXAMPLES[@]}"; do
#     # Training notebooks
#     for notebook in "$SRC_DIR"/$(basename "$ex")/kim*.ipynb; do
#         if [ -f "$notebook" ]; then
#             ln -sf "$notebook" "$DEST_DIR/$(basename "$ex")/$(basename "$notebook")"
#             echo "Symlinked: $notebook -> $DEST_DIR/$(basename "$ex")/$(basename "$notebook")"
#         fi
#     done

#     # Postprocessing notebooks
#     for notebook in "$SRC_DIR"/$(basename "$ex")/postprocess*.ipynb; do
#         if [ -f "$notebook" ]; then
#             ln -sf "$notebook" "$DEST_DIR/$(basename "$ex")/$(basename "$notebook")"
#             echo "Symlinked: $notebook -> $DEST_DIR/$(basename "$ex")/$(basename "$notebook")"
#         fi
#     done
# done