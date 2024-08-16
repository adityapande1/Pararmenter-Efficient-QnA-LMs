#!/usr/bin/env sh
# This script downloads the Stanford CoreNLP models.

CORENLP=stanford-corenlp-4.5.5
SPICELIB=/home/vivek.trivedi/anaconda3/envs/vlt5/lib/python3.8/site-packages/language_evaluation/coco_caption_py3/pycocoevalcap/spice/lib

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

wget http://nlp.stanford.edu/software/$CORENLP.zip

echo "Unzipping..."

unzip $CORENLP.zip -d $SPICELIB/
mv $SPICELIB/$CORENLP/stanford-corenlp-4.5.5.jar $SPICELIB/
mv $SPICELIB/$CORENLP/stanford-corenlp-4.5.5-models.jar $SPICELIB/
rm -f stanford-corenlp-4.5.5.zip
rm -rf $SPICELIB/$CORENLP/

echo "Done."
