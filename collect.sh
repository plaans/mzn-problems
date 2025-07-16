#!/bin/bash
# Bash script to collect all error messages of a list of file.
# INSTANCES_FILE must contain one instances per line.
# All error messages are written in OUTPUT_FILE.

if test $# -ne 2
then
    echo "usage: collect.sh INSTANCES_FILE OUTPUT_FILE" 1>&2
    exit 1
fi

INSTANCES_FILE=$1
OUTPUT_FILE=$2

ERROR_FILE=temp/error.txt

if ! test -f $INSTANCES_FILE
then
    echo "no file $INSTANCES_FILE" 1>&2
    exit 2
fi

INSTANCES=$(cat $INSTANCES_FILE)
echo -n "" > $OUTPUT_FILE

for INSTANCE in $INSTANCES
do
    python mznb.py test $INSTANCE > /dev/null

    if test -f $ERROR_FILE
    then
        echo "--------------------" >> $OUTPUT_FILE
        echo $INSTANCE >> $OUTPUT_FILE
        cat $ERROR_FILE >> $OUTPUT_FILE
    fi
done