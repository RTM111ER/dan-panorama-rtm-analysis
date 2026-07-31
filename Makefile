.PHONY: test quick heavy hashes verify-hashes

test:
	python -m unittest discover -s tests -v

quick:
	bash scripts/run_quick.sh

heavy:
	bash scripts/run_heavy.sh

hashes:
	find results/precomputed -maxdepth 1 -type f ! -name 'SHA256SUMS.txt' -print0 | sort -z | xargs -0 sha256sum > results/precomputed/SHA256SUMS.txt

verify-hashes:
	sha256sum -c results/precomputed/SHA256SUMS.txt
