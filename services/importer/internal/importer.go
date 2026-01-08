package importer

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"github.com/segmentio/encoding/json"
	"github.com/turanic/gs_search/pkg/store"
	"golang.org/x/sync/errgroup"
)

// Vectorizer is an interface for generating embeddings from text.
type Vectorizer interface {
	VectorizeBatch(texts []string) ([][]byte, error)
}

// Store is an interface for storing and retrieving articles.
type Store interface {
	CreateVectorIndex(ctx context.Context) error
	StoreArticles(ctx context.Context, articles []store.Article) error
}

// Importer represents the service importing articles from a target source.
type Importer struct {
	target           string
	store            Store
	vectorizerClient Vectorizer
	logger           *slog.Logger
	maxGoroutines    int
	httpClient       *http.Client
}

// New creates a new Importer instance.
func New(url string, store Store, vectorizerClient Vectorizer, logger *slog.Logger, maxGoroutines int) *Importer {
	return &Importer{
		target:           url,
		store:            store,
		vectorizerClient: vectorizerClient,
		logger:           logger,
		maxGoroutines:    maxGoroutines,
		httpClient:       &http.Client{Timeout: 10 * time.Second},
	}
}

func (i *Importer) Start(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	if err := i.store.CreateVectorIndex(ctx); err != nil {
		i.logger.Error("Failed to create index", "error", err)
	}

	if err := i.initialImport(ctx); err != nil {
		i.logger.Error("Failed to perform initial import", "error", err)
	}

	i.logger.Info("Starting iterative pulling", "target", i.target, "interval", interval.Seconds())
	for {
		select {
		case <-ticker.C:
			// TODO: make number of posts configurable.
			i.logger.Info("Pulling 20 latest posts", "target", i.target)
			_, err := i.vectorizePostsPage(ctx, 1, 20)
			if err != nil {
				i.logger.Error("Failed to pull last posts", "error", err)
			}
		case <-ctx.Done():
			i.logger.Info("Stopping importer", "feed", i.target)
			return
		}
	}
}

// Article represents the data from a website post.
type Article struct {
	Title       string
	Description string
	Link        string
}

// VectorizedArticle represents an article along with its embedding.
type VectorizedArticle struct {
	Article   Article
	Embedding []byte
}

// initialImport fetches and store an initial import of articles from the sitemap.
func (i *Importer) initialImport(ctx context.Context) error {
	// First page is fetched without concurrency to get the total number of pages.
	nbPages, err := i.vectorizePostsPage(ctx, 1, 100)
	if err != nil {
		return fmt.Errorf("failed to fetch number of pages: %w", err)
	}

	grp := errgroup.Group{}
	grp.SetLimit(i.maxGoroutines)
	for page := 2; page <= nbPages; page++ {
		page := page // Capture loop variable.
		grp.Go(func() error {
			_, err := i.vectorizePostsPage(ctx, page, 100)
			return err
		})
		if page%10 == 0 && nbPages > 0 {
			percentageCompletion := (page * 100) / nbPages
			i.logger.Info("Initial import progress", "completion_percent", percentageCompletion)
		}
	}

	if err := grp.Wait(); err != nil {
		return fmt.Errorf("error during initial import: %w", err)
	}
	i.logger.Info("Initial import completed")

	return nil
}

// vectorizeArticle generates the embeddings for given articles.
func (i *Importer) vectorizeArticles(articles []Article) ([]VectorizedArticle, error) {
	textsToVectorize := make([]string, 0, len(articles))
	for _, article := range articles {
		if article.Description == "" {
			i.logger.Warn("Article has empty description, vectorizing only the title", "title", article.Title)
		}
		textsToVectorize = append(textsToVectorize, fmt.Sprintf("%s. %s", article.Title, article.Description))
	}

	// Use batch vectorization - sends all texts in a single request
	embeddings, err := i.vectorizerClient.VectorizeBatch(textsToVectorize)
	if err != nil {
		return nil, fmt.Errorf("failed to vectorize articles in batch: %w", err)
	}

	vectorizedArticles := make([]VectorizedArticle, 0, len(articles))
	for i, embedding := range embeddings {
		vectorizedArticles = append(vectorizedArticles, VectorizedArticle{
			Article:   articles[i],
			Embedding: embedding,
		})
	}

	i.logger.Debug("Successfully vectorized articles", "count", len(vectorizedArticles))
	return vectorizedArticles, nil
}

// storeArticles stores the vectorized articles.
func (i *Importer) storeArticles(ctx context.Context, vectorizedArticles []VectorizedArticle) error {
	articles := make([]store.Article, 0, len(vectorizedArticles))
	for _, va := range vectorizedArticles {
		articles = append(articles, store.Article{
			Title:     va.Article.Title,
			Link:      va.Article.Link,
			Embedding: va.Embedding,
		})
	}

	if err := i.store.StoreArticles(ctx, articles); err != nil {
		i.logger.Error("Failed to store articles", "error", err)
		return fmt.Errorf("failed to store articles: %w", err)
	}

	i.logger.Debug("Successfully stored articles", "count", len(vectorizedArticles))
	return nil
}

// WPPost represents a WordPress post structure from the REST API.
type WPPost struct {
	Title   map[string]interface{} `json:"title"`
	Excerpt map[string]interface{} `json:"excerpt"`
	Link    string                 `json:"link"`
}

func (i *Importer) vectorizePostsPage(ctx context.Context, page, hitsPerPage int) (int, error) {
	posts, nbPages, err := i.fetchPostsPage(page, hitsPerPage)
	if err != nil {
		return 0, fmt.Errorf("failed to fetch posts for page %d: %w", page, err)
	}

	articles := make([]Article, 0, len(posts))
	for _, post := range posts {
		title, _ := post.Title["rendered"].(string)
		excerpt, _ := post.Excerpt["rendered"].(string)
		articles = append(articles, Article{
			Title:       title,
			Description: excerpt,
			Link:        post.Link,
		})
	}
	i.logger.Debug("Fetched page", "page", page, "article_count", len(posts), "max_pages", nbPages)

	vectorizedArticles, err := i.vectorizeArticles(articles)
	if err != nil {
		return 0, fmt.Errorf("failed to vectorize articles: %w", err)
	}

	if err := i.storeArticles(ctx, vectorizedArticles); err != nil {
		return 0, fmt.Errorf("failed to store articles: %w", err)
	}
	return nbPages, nil
}

func (i *Importer) fetchPostsPage(page, hitsPerPage int) ([]WPPost, int, error) {
	url := fmt.Sprintf("%s/wp-json/wp/v2/posts?_fields=title,excerpt,link&per_page=%d&page=%d", i.target, hitsPerPage, page)
	resp, err := i.httpClient.Get(url)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to create request: %w", err)
	}
	defer resp.Body.Close()

	nbPagesHeader := resp.Header.Get("X-WP-TotalPages")
	nbPages, err := strconv.Atoi(nbPagesHeader)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to parse total pages: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, 0, fmt.Errorf("failed to fetch posts, status code: %d", resp.StatusCode)
	}

	var posts []WPPost
	if err := json.NewDecoder(resp.Body).Decode(&posts); err != nil {
		return nil, 0, fmt.Errorf("failed to decode response: %w", err)
	}

	return posts, nbPages, nil
}
