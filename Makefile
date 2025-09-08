.PHONY: build test run-selftest run-all

build:
\tdocker build -t egr-rc .

test:
\tpytest -q

run-selftest:
\tdocker run --rm -v "$(PWD)/out:/work/out" egr-rc egr-rc-selftest --outdir /work/out

run-all:
\tdocker run --rm \
\t  -v "$(PWD)/data:/work/data" \
\t  -v "$(PWD)/out:/work/out" \
\t  egr-rc egr-rc-run-all --data /work/data/SPARC_MassModels.csv --outdir /work/out
