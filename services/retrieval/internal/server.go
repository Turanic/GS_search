package retrieval

import (
	"context"
	"log/slog"
	"net/http"
	"time"

	"github.com/segmentio/encoding/json"
	"github.com/turanic/gs_search/pkg/store"
)

// Vectorizer is an interface for generating embeddings from text.
type Vectorizer interface {
	Vectorize(text string) ([]byte, error)
}

// Store is an interface for performing vector search operations.
type Store interface {
	VectorSearch(ctx context.Context, queryEmbedding []byte, k int) ([]store.SearchHit, error)
	Close() error
}

// Server represents the retrieval service server.
type Server struct {
	serverPort       string
	httpServer       *http.Server
	store            Store
	vectorizerClient Vectorizer
	logger           *slog.Logger
}

// New creates a new retrieval server instance.
func New(serverPort string, store Store, vectorizerClient Vectorizer, logger *slog.Logger) (*Server, error) {
	return &Server{
		serverPort:       serverPort,
		store:            store,
		vectorizerClient: vectorizerClient,
		logger:           logger,
	}, nil
}

// Start starts the HTTP server for the retrieval service. The call is blocking.
func (s *Server) Start() error {
	mux := http.NewServeMux()
	mux.HandleFunc("/search", s.handleSearch)
	mux.HandleFunc("/health", s.handleHealth)

	s.httpServer = &http.Server{
		Addr:         ":" + s.serverPort,
		Handler:      mux,
		ReadTimeout:  1 * time.Second,
		WriteTimeout: 1 * time.Second,
	}

	s.logger.Info("Retrieval service starting", "port", s.serverPort)
	return s.httpServer.ListenAndServe()
}

// Shutdown gracefully shuts down the server.
func (s *Server) Shutdown() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := s.httpServer.Shutdown(ctx); err != nil {
		s.logger.Error("Error shutting down server", "error", err)
	}
	s.store.Close()
}

// SearchRequest represents a search request payload.
type SearchRequest struct {
	Query string `json:"query"`
}

// SearchResult represents a single search result.
type SearchResult struct {
	Title string  `json:"title"`
	URL   string  `json:"url"`
	Score float64 `json:"score"`
}

// SearchResponse represents the search response payload.
type SearchResponse struct {
	Results []SearchResult `json:"results"`
	Count   int            `json:"count"`
}

// handleSearch handles the /search endpoint.
func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}
	s.logger.Debug("Search query received", "query", req.Query)

	embeddingBytes, err := s.vectorizerClient.Vectorize(req.Query)
	if err != nil {
		s.logger.Error("Failed to generate query embedding", "error", err, "query", req.Query)
		http.Error(w, "Failed to generate query embedding", http.StatusInternalServerError)
		return
	}

	// TODO: Make 'k' configurable via request parameters.
	searchResults, err := s.store.VectorSearch(context.Background(), embeddingBytes, 10)
	if err != nil {
		s.logger.Error("Vector search failed", "error", err, "query", req.Query)
		http.Error(w, "Vector search failed", http.StatusInternalServerError)
		return
	}

	results := make([]SearchResult, 0, len(searchResults))
	for _, sr := range searchResults {
		results = append(results, SearchResult{
			Title: sr.Title,
			URL:   sr.Link,
			Score: sr.Score,
		})
	}

	response := SearchResponse{
		Results: results,
		Count:   len(results),
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		s.logger.Error("Failed to encode search response", "error", err)
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		return
	}
}

// handleHealth handles the /health endpoint.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	s.logger.Debug("Received health check request")
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]string{"status": "ok"}); err != nil {
		s.logger.Error("Failed to encode health response", "error", err)
	}
}
