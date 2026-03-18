#!/bin/bash

SCRIPT_DIR=$(dirname -- "$0")
cd "${SCRIPT_DIR}" || exit 1

grep -E '^https?://' ./urls.txt | xargs -n 1 wget
