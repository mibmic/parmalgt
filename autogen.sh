#!/bin/bash
rm -f configure
mkdir -p m4
autoreconf --install --force || exit 1
