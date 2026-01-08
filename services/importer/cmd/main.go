package main

import (
	"context"
	"log"
	"log/slog"
	"os"
	"time"

	"github.com/kelseyhightower/envconfig"
	"github.com/turanic/gs_search/pkg/store"
	"github.com/turanic/gs_search/pkg/vectorization"
	importer "github.com/turanic/gs_search/services/importer/internal"
)

// Config holds the configuration for the importer service.
type Config struct {
	RedisAddr           string        `envconfig:"REDIS_ADDR"`
	RedisPassword       string        `envconfig:"REDIS_PASSWORD"`
	VectorizerAddr      string        `envconfig:"VECTORIZER_ADDR"`
	TargetURL           string        `envconfig:"TARGET_URL"`
	PollInterval        time.Duration `envconfig:"POLL_INTERVAL" default:"10s"`
	DebugMode           bool          `envconfig:"DEBUG_MODE" default:"false"`
	EmbeddingDimension  int           `envconfig:"EMBEDDING_DIMENSION" default:"384"`
	ImportMaxGoroutines int           `envconfig:"IMPORT_MAX_GOROUTINES" default:"1"`
}

func main() {
	var config Config
	if err := envconfig.Process("", &config); err != nil {
		log.Fatalf("Failed to process config: %v", err)
	}

	// Initialize logger based on debug mode
	logLevel := slog.LevelInfo
	if config.DebugMode {
		logLevel = slog.LevelDebug
	}
	handler := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: logLevel,
	})
	logger := slog.New(handler).With("service", "importer")

	redisClient := store.New(config.RedisAddr, config.RedisPassword, config.EmbeddingDimension)
	if err := redisClient.Ping(context.Background()); err != nil {
		logger.Error("Failed to connect to Redis", "error", err)
		log.Fatalf("Failed to connect to Redis: %v", err)
	}

	vectorizerClient := vectorization.New(config.VectorizerAddr)
	if err := vectorizerClient.HealthCheck(); err != nil {
		logger.Warn("Vectorizer health check failed", "error", err)
	}

	// TODO: handle graceful shutdown. Not critical for the importer as it does not serve requests...
	i := importer.New(config.TargetURL, redisClient, vectorizerClient, logger, config.ImportMaxGoroutines)
	i.Start(context.Background(), config.PollInterval)
}
