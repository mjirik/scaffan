#!/bin/bash

# nastaveni domovskeho adresare, v promenne $LOGNAME je ulozeno vase prihlasovaci jmeno
DATADIR="/storage/plzen1/home/$LOGNAME/"
# nebo snad "/storage/plzen4-ntis/home/$LOGNAME/"  ?

# nacteni aplikacniho modulu, ktery zpristupni aplikaci Gaussian verze 3
# module add g03

# nastaveni automatickeho vymazani adresare SCRATCH pro pripad chyby pri behu ulohy
trap 'clean_scratch' TERM EXIT

# vstup do adresare SCRATCH, nebo v pripade neuspechu ukonceni s chybovou hodnotou rovnou 1
cd $SCRATCHDIR || exit 1

# priprava vstupnich dat (kopirovani dat na vypocetni uzel)
# cp $DATADIR/gaussian_test.com $SCRATCHDIR

# spusteni aplikace - samotny vypocet
export PATH=/storage/plzen1/home/$LOGNAME/miniconda3/bin:$PATH
source activate scaffan
julia /storage/plzen1/home/$LOGNAME/projects/scaffan/experiments/lobulus_precision.py > results.out


# kopirovani vystupnich dat z vypocetnicho uzlu do domovskeho adresare,
# pokud by pri kopirovani doslo k chybe, nebude adresar SCRATCH vymazan pro moznost rucniho vyzvednuti dat
cp results.out $DATADIR || export CLEAN_SCRATCH=false