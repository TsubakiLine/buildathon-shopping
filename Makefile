IMAGE ?= shopping-sessions
DEV_DATA ?= ./dev-data
DEV_ANSWER ?= ./dev-answer
DEV_OUT ?= ./dev-out
NCPU ?= 8

.PHONY: build docker-build fixtures check-dev clean

build:
	RUSTFLAGS="-C target-cpu=native" cargo build --release

docker-build:
	docker build --platform linux/amd64 -t $(IMAGE) .

fixtures: build
	@set -e; \
	for dir in fixtures/*/; do \
		name=$$(basename "$$dir"); \
		rm -rf /tmp/ss-fix-out; \
		DATA_DIR="$$dir/input" OUTPUT_DIR=/tmp/ss-fix-out NCPU=$(NCPU) ./target/release/shopping-sessions; \
		python3 scripts/compare.py /tmp/ss-fix-out "$$dir/expected" >/dev/null || { echo "FAIL $$name"; python3 scripts/compare.py /tmp/ss-fix-out "$$dir/expected"; exit 1; }; \
		echo "PASS $$name"; \
	done

check-dev: build
	rm -rf $(DEV_OUT)
	DATA_DIR=$(DEV_DATA) OUTPUT_DIR=$(DEV_OUT) NCPU=$(NCPU) ./target/release/shopping-sessions
	python3 scripts/compare.py $(DEV_OUT) $(DEV_ANSWER)

docker-check-dev: docker-build
	rm -rf $(DEV_OUT) && mkdir -p $(DEV_OUT) /tmp/ss-scratch
	docker run --rm --platform linux/amd64 --cpus=8 --memory=24g --memory-swap=24g \
		--network=none --read-only --tmpfs /tmp:size=64m \
		-v "$$(realpath $(DEV_DATA)):/data:ro" \
		-v "$$(realpath $(DEV_OUT)):/output" \
		-v "/tmp/ss-scratch:/scratch" \
		-e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/scratch \
		-e HOME=/scratch -e NCPU=8 -e MEM_GB=24 \
		$(IMAGE)
	python3 scripts/compare.py $(DEV_OUT) $(DEV_ANSWER)

clean:
	rm -rf $(DEV_OUT) /tmp/ss-fix-out /tmp/ss-scratch