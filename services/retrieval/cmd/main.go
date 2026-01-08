package main

import (
	"context"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/kelseyhightower/envconfig"
	"github.com/turanic/gs_search/pkg/store"
	"github.com/turanic/gs_search/pkg/vectorization"
	retrieval "github.com/turanic/gs_search/services/retrieval/internal"
)

// Config holds the configuration for the retrieval service.
type Config struct {
	RedisAddr          string `envconfig:"REDIS_ADDR"`
	RedisPassword      string `envconfig:"REDIS_PASSWORD"`
	ServerPort         string `envconfig:"SERVER_PORT" default:"8080"`
	VectorizerAddr     string `envconfig:"VECTORIZER_ADDR"`
	DebugMode          bool   `envconfig:"DEBUG_MODE" default:"false"`
	EmbeddingDimension int    `envconfig:"EMBEDDING_DIMENSION" default:"384"`
}

func main() {
	var config Config
	if err := envconfig.Process("", &config); err != nil {
		log.Fatalf("Failed to process config: %v", err)
	}

	logLevel := slog.LevelInfo
	if config.DebugMode {
		logLevel = slog.LevelDebug
	}
	handler := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: logLevel,
	})
	logger := slog.New(handler).With("service", "retrieval")

	redisClient := store.New(config.RedisAddr, config.RedisPassword, config.EmbeddingDimension)
	if err := redisClient.Ping(context.Background()); err != nil {
		logger.Error("Failed to connect to Redis", "error", err)
		log.Fatalf("Failed to connect to Redis: %v", err)
	}

	vectorizerClient := vectorization.New(config.VectorizerAddr)
	if err := vectorizerClient.HealthCheck(); err != nil {
		logger.Warn("Vectorizer health check failed", "error", err)
	}

	srv, err := retrieval.New(config.ServerPort, redisClient, vectorizerClient, logger)
	if err != nil {
		log.Fatalf("Failed to initialize retrieval service: %v", err)
	}

	go func() {
		if err := srv.Start(); err != nil {
			logger.Error("Failed to start retrieval service", "error", err)
		}
	}()

	// Setup graceful shutdown.
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan
	logger.Info("Shutting down retrieval service")
	srv.Shutdown()
	logger.Info("Retrieval service stopped")
}
