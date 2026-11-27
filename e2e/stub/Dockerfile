FROM golang:1.26 AS build
WORKDIR /src
COPY go.mod go.sum ./
COPY e2e/stub/main.go e2e/stub/main.go
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /stub ./e2e/stub

FROM debian:bookworm-slim
COPY --from=build /stub /stub
EXPOSE 9123
HEALTHCHECK --interval=1s --timeout=2s --retries=30 CMD ["/stub", "-healthcheck"]
ENTRYPOINT ["/stub"]
