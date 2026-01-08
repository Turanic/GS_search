package importer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/turanic/gs_search/pkg/store"
)

// mockVectorizer implements the Vectorizer interface for testing.
type mockVectorizer struct {
	embeddings [][]byte
	err        error
}

func (m *mockVectorizer) VectorizeBatch(texts []string) ([][]byte, error) {
	if m.err != nil {
		return nil, m.err
	}
	result := make([][]byte, len(texts))
	for i := range texts {
		if i < len(m.embeddings) {
			result[i] = m.embeddings[i]
		} else {
			result[i] = []byte(fmt.Sprintf("embedding-%d", i))
		}
	}
	return result, nil
}

// mockStore implements the Store interface for testing.
type mockStore struct {
	storeErr       error
	indexErr       error
	mtx            sync.Mutex // Protects storedArticles.
	storedArticles []store.Article
}

func (m *mockStore) CreateVectorIndex(ctx context.Context) error {
	return m.indexErr
}

func (m *mockStore) StoreArticles(ctx context.Context, articles []store.Article) error {
	if m.storeErr != nil {
		return m.storeErr
	}
	m.mtx.Lock()
	defer m.mtx.Unlock()
	m.storedArticles = append(m.storedArticles, articles...)
	return nil
}

// buildWordPressResponse converts articles to WordPress post format and writes JSON response.
func buildWordPressResponse(w http.ResponseWriter, articles []Article) error {
	var posts []map[string]interface{}
	for _, article := range articles {
		posts = append(posts, map[string]interface{}{
			"title":   map[string]string{"rendered": article.Title},
			"excerpt": map[string]string{"rendered": article.Description},
			"link":    article.Link,
		})
	}

	jsonData, err := json.Marshal(posts)
	if err != nil {
		return err
	}
	_, err = w.Write(jsonData)
	return err
}

// wpServerMock mocks a WordPress REST API server for testing.
type wpServerMock struct {
	t             *testing.T
	articles      []Article
	totalPages    int
	statusCode    int
	invalidJSON   bool
	missingHeader bool
	server        *httptest.Server
	mtx           sync.Mutex // Protects pageRequests.
	pageRequests  int
}

// Listen starts the mock WordPress server and returns the server instance.
func (w *wpServerMock) Listen() *httptest.Server {
	w.server = httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		pageParam := r.URL.Query().Get("page")
		pageNumber := 1
		if pageParam != "" {
			_, err := fmt.Sscanf(pageParam, "%d", &pageNumber)
			if err != nil {
				pageNumber = 1
			}
		}

		w.mtx.Lock()
		w.pageRequests++
		w.mtx.Unlock()

		if !w.missingHeader {
			rw.Header().Set("X-WP-TotalPages", fmt.Sprintf("%d", w.totalPages))
		}
		rw.Header().Set("Content-Type", "application/json")
		rw.WriteHeader(w.statusCode)

		if w.statusCode != http.StatusOK {
			_, err := io.WriteString(rw, http.StatusText(w.statusCode))
			require.NoError(w.t, err)
			return
		}

		if w.invalidJSON {
			_, err := io.WriteString(rw, "Invalid JSON")
			require.NoError(w.t, err)
			return
		}

		err := buildWordPressResponse(rw, w.articles)
		require.NoError(w.t, err)
	}))
	return w.server
}

func TestVectorizePostsPage(t *testing.T) {
	t.Parallel()
	testCases := []struct {
		name               string
		articles           []Article
		totalPages         int
		statusCode         int
		invalidJSON        bool
		missingHeader      bool
		mockVectorizer     *mockVectorizer
		mockStore          *mockStore
		expectError        bool
		errorContains      string
		expectedPages      int
		expectedArticles   int
		expectedTitles     []string
		validateEmbeddings bool
	}{
		{
			name: "Success",
			articles: []Article{
				{
					Title:       "Test Article 1",
					Description: "This is the first test article",
					Link:        "https://example.com/article1",
				},
				{
					Title:       "Test Article 2",
					Description: "This is the second test article",
					Link:        "https://example.com/article2",
				},
			},
			totalPages: 5,
			statusCode: http.StatusOK,
			mockVectorizer: &mockVectorizer{
				embeddings: [][]byte{
					[]byte("embedding1"),
					[]byte("embedding2"),
				},
			},
			mockStore:          &mockStore{},
			expectError:        false,
			expectedPages:      5,
			expectedArticles:   2,
			expectedTitles:     []string{"Test Article 1", "Test Article 2"},
			validateEmbeddings: true,
		},
		{
			name:           "HTTPError",
			articles:       nil,
			totalPages:     1,
			statusCode:     http.StatusInternalServerError,
			mockVectorizer: &mockVectorizer{},
			mockStore:      &mockStore{},
			expectError:    true,
			errorContains:  "failed to fetch posts",
		},
		{
			name: "VectorizerError",
			articles: []Article{
				{Title: "Test", Description: "Content", Link: "http://test.com"},
			},
			totalPages: 1,
			statusCode: http.StatusOK,
			mockVectorizer: &mockVectorizer{
				err: errors.New("vectorization failed"),
			},
			mockStore:     &mockStore{},
			expectError:   true,
			errorContains: "failed to vectorize articles",
		},
		{
			name: "StoreError",
			articles: []Article{
				{Title: "Test", Description: "Content", Link: "http://test.com"},
			},
			totalPages: 1,
			statusCode: http.StatusOK,
			mockVectorizer: &mockVectorizer{
				embeddings: [][]byte{[]byte("embedding")},
			},
			mockStore: &mockStore{
				storeErr: errors.New("database connection failed"),
			},
			expectError:   true,
			errorContains: "failed to store articles",
		},
		{
			name:             "EmptyResponse",
			articles:         []Article{},
			totalPages:       0,
			statusCode:       http.StatusOK,
			mockVectorizer:   &mockVectorizer{},
			mockStore:        &mockStore{},
			expectError:      false,
			expectedPages:    0,
			expectedArticles: 0,
		},
		{
			name:           "InvalidJSON",
			articles:       nil,
			totalPages:     1,
			statusCode:     http.StatusOK,
			invalidJSON:    true,
			mockVectorizer: &mockVectorizer{},
			mockStore:      &mockStore{},
			expectError:    true,
			errorContains:  "failed to fetch posts",
		},
		{
			name:           "MissingPagesHeader",
			articles:       []Article{},
			totalPages:     0,
			statusCode:     http.StatusOK,
			missingHeader:  true,
			mockVectorizer: &mockVectorizer{},
			mockStore:      &mockStore{},
			expectError:    true,
			errorContains:  "failed to parse total pages",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mock := &wpServerMock{
				t:             t,
				articles:      tc.articles,
				totalPages:    tc.totalPages,
				statusCode:    tc.statusCode,
				invalidJSON:   tc.invalidJSON,
				missingHeader: tc.missingHeader,
			}
			server := mock.Listen()
			defer server.Close()

			logger := slog.New(slog.NewTextHandler(io.Discard, nil))
			importer := New(server.URL, tc.mockStore, tc.mockVectorizer, logger, 5)
			nbPages, err := importer.vectorizePostsPage(context.Background(), 1, 10)

			if tc.expectError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			assert.Equal(t, nbPages, tc.expectedPages)

			// Validate stored articles in the mocked store.
			assert.Equal(t, len(tc.mockStore.storedArticles), tc.expectedArticles)
			for i, storedArticle := range tc.mockStore.storedArticles {
				assert.Equal(t, storedArticle.Title, tc.expectedTitles[i])
				if tc.validateEmbeddings {
					assert.Equal(t, storedArticle.Embedding, tc.mockVectorizer.embeddings[i])
				}
			}
		})
	}
}

func TestInitialImport(t *testing.T) {
	t.Parallel()
	testCases := []struct {
		name               string
		articlesPerPage    []Article
		totalPages         int
		statusCode         int
		mockVectorizer     *mockVectorizer
		mockStore          *mockStore
		maxGoroutines      int
		expectError        bool
		errorContains      string
		expectedTotalCalls int
	}{
		{
			name: "SuccessSinglePage",
			articlesPerPage: []Article{
				{Title: "Article 1", Description: "Desc 1", Link: "http://test.com/1"},
				{Title: "Article 2", Description: "Desc 2", Link: "http://test.com/2"},
			},
			totalPages: 1,
			statusCode: http.StatusOK,
			mockVectorizer: &mockVectorizer{
				embeddings: [][]byte{[]byte("emb1"), []byte("emb2")},
			},
			mockStore:          &mockStore{},
			maxGoroutines:      5,
			expectError:        false,
			expectedTotalCalls: 2, // 2 articles on 1 page
		},
		{
			name: "SuccessMultiplePages",
			articlesPerPage: []Article{
				{Title: "Article 1", Description: "Desc 1", Link: "http://test.com/1"},
			},
			totalPages: 3,
			statusCode: http.StatusOK,
			mockVectorizer: &mockVectorizer{
				embeddings: [][]byte{[]byte("emb1")},
			},
			mockStore:          &mockStore{},
			maxGoroutines:      1,
			expectError:        false,
			expectedTotalCalls: 3, // 1 article per page * 3 pages
		},
		{
			name: "SuccessConcurrentPages",
			articlesPerPage: []Article{
				{Title: "Article 1", Description: "Desc 1", Link: "http://test.com/1"},
			},
			totalPages: 4,
			statusCode: http.StatusOK,
			mockVectorizer: &mockVectorizer{
				embeddings: [][]byte{[]byte("emb1")},
			},
			mockStore:          &mockStore{},
			maxGoroutines:      4,
			expectError:        false,
			expectedTotalCalls: 4, // 1 article per page * 4 pages
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mock := &wpServerMock{
				t:          t,
				articles:   tc.articlesPerPage,
				totalPages: tc.totalPages,
				statusCode: tc.statusCode,
			}
			server := mock.Listen()
			defer server.Close()

			logger := slog.New(slog.NewTextHandler(io.Discard, nil))
			importer := New(server.URL, tc.mockStore, tc.mockVectorizer, logger, tc.maxGoroutines)
			err := importer.initialImport(context.Background())
			if tc.expectError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}

			assert.Equal(t, tc.expectedTotalCalls, len(tc.mockStore.storedArticles),
				"Expected %d articles stored, got %d", tc.expectedTotalCalls, len(tc.mockStore.storedArticles))
			assert.Equal(t, tc.totalPages, mock.pageRequests,
				"Expected %d page requests, got %d", tc.totalPages, mock.pageRequests)
		})
	}
}
