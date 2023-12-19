#!/bin/bash

echo 'Getting the EmoryNLP dataset'
git clone https://github.com/emorynlp/character-mining.git
mkdir EmoryNLP
cp -r character-mining/tsv/ ./EmoryNLP
rm -rf character-mining.git

echo '\nGetting the DailyDialog dataset'
curl -OLv 'http://yanran.li/files/ijcnlp_dailydialog.zip' > DailyDialog.zip
unzip DailyDialog.zip 