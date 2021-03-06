#!/bin/bash
#PBS -l select=1:ncpus=1:mem=16gb:scratch_local=10gb
#PBS -l walltime=016:00:00
# modify/delete the above given guidelines according to your job's needs
# Please note that only one select= argument is allowed at a time.

# # PBS -l select=1:ncpus=2:mem=1gb:scratch_local=4gb

# add to qsub with:
# qsub scaffan_experiment_1.sh

# nastaveni domovskeho adresare, v promenne $LOGNAME je ulozeno vase prihlasovaci jmeno
DATADIR="/storage/plzen1/home/$LOGNAME/projects/scaffan/experiments/metacentrum/"
DATADIR="/storage/plzen4-ntis/projects/queetech/medical/processed/scaffan2019/metacentrum"
# nebo snad "/storage/plzen4-ntis/home/$LOGNAME/"  ?

# nacteni aplikacniho modulu, ktery zpristupni aplikaci Gaussian verze 3
# module add g03


SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

# nastaveni automatickeho vymazani adresare SCRATCH pro pripad chyby pri behu ulohy
trap 'clean_scratch' TERM EXIT

# vstup do adresare SCRATCH, nebo v pripade neuspechu ukonceni s chybovou hodnotou rovnou 1
cd $SCRATCHDIR || exit 1

# priprava vstupnich dat (kopirovani dat na vypocetni uzel)
# cp $DATADIR/gaussian_test.com $SCRATCHDIR

# spusteni aplikace - samotny vypocet
export PATH=/storage/plzen1/home/$LOGNAME/miniconda3/bin:$PATH
source activate scaffan
echo /storage/plzen1/home/$LOGNAME/projects/scaffan/experiments/metacentrum/SA_experiments.xlsx
# this is because of python click
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# python -m scaffan set --common-spreadsheet-file /storage/plzen1/home/$LOGNAME/projects/scaffan/experiments/metacentrum/SA_experiments.xlsx
python -m scaffan set --common-spreadsheet-file /storage/plzen4-ntis/projects/queetech/medical/processed/scaffan2019/metacentrum/SA_experiments.xlsx
python -m io3d.datasets -sdp  /storage/plzen4-ntis/projects/queetech/
python /storage/plzen1/home/$LOGNAME/projects/scaffan/experiments/lobulus_seg_eval_2.py > results.out

echo "$DIR"
ls
# kopirovani vystupnich dat z vypocetnicho uzlu do domovskeho adresare,
# pokud by pri kopirovani doslo k chybe, nebude adresar SCRATCH vymazan pro moznost rucniho vyzvednuti dat
cp results.out $DATADIR && cp -r SA_* $DATADIR || export CLEAN_SCRATCH=false
#cp results.out $DATADIR && cp scaffan.log $DATADIR && cp -r test_run_lobuluses_output_dir* $DATADIR || export CLEAN_SCRATCH=false
#cp results.out $DATADIR || export CLEAN_SCRATCH=false
