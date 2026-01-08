package retrieval

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/turanic/gs_search/pkg/store"
)

// mockVectorizer implements the Vectorizer interface for testing.
type mockVectorizer struct {
	embedding []byte
	err       error
}

func (m *mockVectorizer) Vectorize(text string) ([]byte, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.embedding, nil
}

// mockStore implements the Store interface for testing.
type mockStore struct {
	searchResults []store.SearchHit
	searchErr     error
}

func (m *mockStore) VectorSearch(ctx context.Context, queryEmbedding []byte, k int) ([]store.SearchHit, error) {
	if m.searchErr != nil {
		return nil, m.searchErr
	}
	return m.searchResults, nil
}

func (m *mockStore) Close() error {
	return nil
}

func TestHandleSearch(t *testing.T) {
	testCases := []struct {
		name               string
		requestBody        interface{}
		requestMethod      string
		mockVectorizer     *mockVectorizer
		mockStore          *mockStore
		expectedStatus     int
		expectedCount      int
		expectedResults    []SearchResult
		expectErrorMessage string
	}{
		{
			name: "Success",
			requestBody: SearchRequest{
				Query: "test query",
			},
			requestMethod: http.MethodPost,
			mockVectorizer: &mockVectorizer{
				embedding: []byte("test-embedding"),
			},
			mockStore: &mockStore{
				searchResults: []store.SearchHit{
					{Title: "Result 1", Link: "http://example.com/1", Score: 0.95},
					{Title: "Result 2", Link: "http://example.com/2", Score: 0.85},
				},
			},
			expectedStatus: http.StatusOK,
			expectedCount:  2,
			expectedResults: []SearchResult{
				{Title: "Result 1", URL: "http://example.com/1", Score: 0.95},
				{Title: "Result 2", URL: "http://example.com/2", Score: 0.85},
			},
		},
		{
			name: "SuccessEmptyResults",
			requestBody: SearchRequest{
				Query: "no results query",
			},
			requestMethod: http.MethodPost,
			mockVectorizer: &mockVectorizer{
				embedding: []byte("test-embedding"),
			},
			mockStore: &mockStore{
				searchResults: []store.SearchHit{},
			},
			expectedStatus:  http.StatusOK,
			expectedCount:   0,
			expectedResults: []SearchResult{},
		},
		{
			name:               "MethodNotAllowed",
			requestBody:        SearchRequest{Query: "test"},
			requestMethod:      http.MethodGet,
			mockVectorizer:     &mockVectorizer{},
			mockStore:          &mockStore{},
			expectedStatus:     http.StatusMethodNotAllowed,
			expectErrorMessage: "Method not allowed",
		},
		{
			name:               "InvalidRequestBody",
			requestBody:        "invalid json",
			requestMethod:      http.MethodPost,
			mockVectorizer:     &mockVectorizer{},
			mockStore:          &mockStore{},
			expectedStatus:     http.StatusBadRequest,
			expectErrorMessage: "Invalid request body",
		},
		{
			name: "VectorizerError",
			requestBody: SearchRequest{
				Query: "test query",
			},
			requestMethod: http.MethodPost,
			mockVectorizer: &mockVectorizer{
				err: errors.New("vectorization failed"),
			},
			mockStore:          &mockStore{},
			expectedStatus:     http.StatusInternalServerError,
			expectErrorMessage: "Failed to generate query embedding",
		},
		{
			name: "VectorSearchError",
			requestBody: SearchRequest{
				Query: "test query",
			},
			requestMethod: http.MethodPost,
			mockVectorizer: &mockVectorizer{
				embedding: []byte("test-embedding"),
			},
			mockStore: &mockStore{
				searchErr: errors.New("search failed"),
			},
			expectedStatus:     http.StatusInternalServerError,
			expectErrorMessage: "Vector search failed",
		},
		{
			name: "EmptyQuery",
			requestBody: SearchRequest{
				Query: "",
			},
			requestMethod: http.MethodPost,
			mockVectorizer: &mockVectorizer{
				embedding: []byte("test-embedding"),
			},
			mockStore: &mockStore{
				searchResults: []store.SearchHit{},
			},
			expectedStatus:  http.StatusOK,
			expectedCount:   0,
			expectedResults: []SearchResult{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			logger := slog.New(slog.NewTextHandler(io.Discard, nil))
			server := &Server{
				store:            tc.mockStore,
				vectorizerClient: tc.mockVectorizer,
				logger:           logger,
			}

			var reqBody []byte
			var err error
			if str, ok := tc.requestBody.(string); ok {
				reqBody = []byte(str)
			} else {
				reqBody, err = json.Marshal(tc.requestBody)
				require.NoError(t, err)
			}
			req := httptest.NewRequest(tc.requestMethod, "/search", bytes.NewReader(reqBody))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			server.handleSearch(w, req)
			require.Equal(t, tc.expectedStatus, w.Code, "Expected status code %d, got %d", tc.expectedStatus, w.Code)
			if tc.expectedStatus == http.StatusOK {
				var response SearchResponse
				err := json.NewDecoder(w.Body).Decode(&response)
				require.NoError(t, err, "Failed to decode response")

				require.Equal(t, tc.expectedCount, response.Count, "Expected count %d, got %d", tc.expectedCount, response.Count)
				require.Equal(t, len(tc.expectedResults), len(response.Results), "Expected %d results, got %d", len(tc.expectedResults), len(response.Results))

				for i, expectedResult := range tc.expectedResults {
					require.Equal(t, expectedResult.Title, response.Results[i].Title, "Result %d title mismatch", i)
					require.Equal(t, expectedResult.URL, response.Results[i].URL, "Result %d URL mismatch", i)
					require.Equal(t, expectedResult.Score, response.Results[i].Score, "Result %d score mismatch", i)
				}
				require.Equal(t, "application/json", w.Header().Get("Content-Type"))
			}

			if tc.expectErrorMessage != "" {
				bodyStr := w.Body.String()
				require.Contains(t, bodyStr, tc.expectErrorMessage, "Expected error message to contain '%s'", tc.expectErrorMessage)
			}
		})
	}
}
