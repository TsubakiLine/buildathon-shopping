# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 rust:1-bookworm AS builder

WORKDIR /build
COPY Cargo.toml ./
# Cache dependency compilation
RUN mkdir src && echo "fn main() {}" > src/main.rs && \
    RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release && \
    rm -rf src

COPY src ./src
RUN touch src/main.rs && \
    RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release

FROM --platform=linux/amd64 debian:bookworm-slim

WORKDIR /app
COPY --from=builder /build/target/release/shopping-sessions /app/pipeline
COPY run.sh /app/run.sh
RUN chmod 755 /app/pipeline /app/run.sh

CMD ["/app/pipeline"]