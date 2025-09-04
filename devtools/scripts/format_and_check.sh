#!/usr/bin/env bash
set -e -v
ruff format QuantumPioneer
ruff check --fix QuantumPioneer
