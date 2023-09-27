#!/bin/bash

cd "TAU2020MDev"
pwd
for file in *.zip; do
  unzip "${file}" && rm "${file}"
done