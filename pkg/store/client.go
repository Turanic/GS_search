package store

import (
	"context"
	"fmt"
	"log"

	"github.com/redis/go-redis/v9"
)

const (
	IndexName     = "gs_data"
	ArticlePrefix = "article:"
)

// Client wraps a client enabling interactions with the store.
type Client struct {
	*redis.Client
	embeddingDimension int
}

// New creates a new Redis client with the appropriate configuration.
func New(addr, password string, embeddingDimension int) *Client {
	redisClient := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       0,
		Protocol: 2,
	})

	return &Client{
		Client:             redisClient,
		embeddingDimension: embeddingDimension,
	}
}

// Ping tests the connection to Redis.
func (c *Client) Ping(ctx context.Context) error {
	return c.Client.Ping(ctx).Err()
}

// CreateVectorIndex creates the Redis vector search index if it doesn't exist.
func (c *Client) CreateVectorIndex(ctx context.Context) error {
	result := c.FTCreate(ctx, IndexName, &redis.FTCreateOptions{
		OnHash: true,
		Prefix: []interface{}{
			ArticlePrefix,
		},
	},
		&redis.FieldSchema{
			FieldName: "embedding",
			FieldType: redis.SearchFieldTypeVector,
			VectorArgs: &redis.FTVectorArgs{
				FlatOptions: &redis.FTFlatOptions{
					Type:           "FLOAT32",
					Dim:            c.embeddingDimension,
					DistanceMetric: "COSINE",
				},
			},
		},
		&redis.FieldSchema{
			FieldName: "title",
			FieldType: redis.SearchFieldTypeText,
		},
		&redis.FieldSchema{
			FieldName: "link",
			FieldType: redis.SearchFieldTypeText,
		},
	)

	if err := result.Err(); err != nil {
		if err.Error() == "Index already exists" {
			return nil
		}
		return fmt.Errorf("failed to create index: %w", err)
	}

	log.Printf("Successfully created Redis index: %s", IndexName)
	return nil
}

// Article is the model for a stored article.
type Article struct {
	Title     string
	Link      string
	Embedding []byte
}

// StoreArticles stores multiple articles with their embeddings in Redis using a pipeline.
func (c *Client) StoreArticles(ctx context.Context, articles []Article) error {
	if len(articles) == 0 {
		return nil
	}

	pipe := c.Pipeline()
	for _, article := range articles {
		key := fmt.Sprintf("%s%s", ArticlePrefix, article.Title)
		pipe.HSet(ctx, key, map[string]interface{}{
			"title":     article.Title,
			"link":      article.Link,
			"embedding": article.Embedding,
		})
	}

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to store articles: %w", err)
	}

	return nil
}

// SearchHit represents an article search result, from Redis.
type SearchHit struct {
	Title string
	Link  string
	Score float64
}

// VectorSearch performs a search on the store to retrieve articles.
// The search is a KNN search based on the provided query embedding.
func (c *Client) VectorSearch(ctx context.Context, queryEmbedding []byte, k int) ([]SearchHit, error) {
	// KNN query with score alias for sorting
	knnQuery := fmt.Sprintf("*=>[KNN %d @embedding $query_vec AS vector_score]", k)

	searchCmd := c.FTSearchWithArgs(
		ctx,
		IndexName,
		knnQuery,
		&redis.FTSearchOptions{
			SortBy: []redis.FTSearchSortBy{
				{FieldName: "vector_score", Asc: true},
			},
			DialectVersion: 2,
			Params: map[string]interface{}{
				"query_vec": queryEmbedding,
			},
			Return: []redis.FTSearchReturn{
				{FieldName: "title"},
				{FieldName: "link"},
				{FieldName: "vector_score"},
			},
		},
	)

	if err := searchCmd.Err(); err != nil {
		return nil, fmt.Errorf("vector search failed: %w", err)
	}

	searchResult, err := searchCmd.Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get search results: %w", err)
	}

	results := make([]SearchHit, 0, len(searchResult.Docs))
	for _, doc := range searchResult.Docs {
		title := doc.Fields["title"]
		link := doc.Fields["link"]

		score := 0.0
		if scoreVal := doc.Fields["vector_score"]; scoreVal != "" {
			if _, err := fmt.Sscanf(scoreVal, "%f", &score); err != nil {
				log.Printf("Error parsing score: %v", err)
			}
		}

		results = append(results, SearchHit{
			Title: title,
			Link:  link,
			Score: score,
		})
	}

	return results, nil
}
