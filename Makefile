.PHONY: test quick heavy hashes

test:
	python -m unittest discover -s tests -v

quick:
	bash scripts/run_quick.sh

heavy:
	bash scripts/run_heavy.sh

hashes:
	sha256sum results/precomputed/* > results/precomputed/SHA256SUMS.txt
