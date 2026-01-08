# Build stage
FROM golang:alpine AS builder

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the retrieval binary
RUN CGO_ENABLED=0 GOOS=linux go build -o /retrieval ./services/retrieval/cmd

# Final stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates

WORKDIR /root/

# Copy the binary from builder
COPY --from=builder /retrieval .

EXPOSE 8080

CMD ["./retrieval"]
