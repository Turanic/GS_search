package vectorization

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/segmentio/encoding/json"
)

// Client is a shared HTTP client for the Vectorizer service.
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// EmbeddingData represents a single embedding from the vectorizer.
type EmbeddingData struct {
	Embedding string `json:"embedding"`
	Dimension int    `json:"dimension"`
}

// Response represents the response from the Vectorizer service.
type Response struct {
	Embeddings []EmbeddingData `json:"embeddings"`
}

// New creates a new Vectorizer client.
// TODO: Improve http client configuration.
func New(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// HealthCheck checks if the Vectorizer service is available.
func (c *Client) HealthCheck() error {
	url := fmt.Sprintf("%s/health", c.baseURL)
	resp, err := c.httpClient.Get(url)
	if err != nil {
		return fmt.Errorf("failed to reach vectorizer: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("vectorizer health check failed with status %d", resp.StatusCode)
	}

	return nil
}

// VectorizeBatch generates embedding vectors for a batch of texts.
// The returned embeddings are in the same order as the input texts.
// This method sends all texts to the vectorizer in a single request.
func (c *Client) VectorizeBatch(texts []string) ([][]byte, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("texts cannot be empty")
	}

	reqBody := map[string][]string{"texts": texts}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/embed", c.baseURL)
	resp, err := c.httpClient.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to send request to vectorizer: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("vectorizer returned status %d: %s", resp.StatusCode, string(body))
	}

	rBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read vectorizer response: %w", err)
	}

	var vecResp Response
	if err := json.Unmarshal(rBody, &vecResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal vectorizer response: %w", err)
	}

	if len(vecResp.Embeddings) != len(texts) {
		return nil, fmt.Errorf("expected %d embeddings, got %d", len(texts), len(vecResp.Embeddings))
	}

	embeddings := make([][]byte, len(vecResp.Embeddings))
	for i, embData := range vecResp.Embeddings {
		// Decode base64 embedding bytes
		embeddingBytes, err := base64.StdEncoding.DecodeString(embData.Embedding)
		if err != nil {
			return nil, fmt.Errorf("failed to decode embedding bytes at index %d: %w", i, err)
		}

		if len(embeddingBytes) == 0 {
			return nil, fmt.Errorf("received empty embedding at index %d from vectorizer", i)
		}

		embeddings[i] = embeddingBytes
	}

	return embeddings, nil
}

// Vectorize generates an embedding vector for the given text.
// This is a convenience method that calls VectorizeBatch with a single text.
func (c *Client) Vectorize(text string) ([]byte, error) {
	if text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}

	embeddings, err := c.VectorizeBatch([]string{text})
	if err != nil {
		return nil, err
	}

	return embeddings[0], nil
}
