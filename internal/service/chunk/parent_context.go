package chunk

import (
	"context"
	"fmt"

	"ragflow/internal/engine/types"
)

// invalidateParentContexts removes derived parent text before a source mutation.
// A parent can contain text from several children, including children that did
// not match the query. Retaining it after an edit, disable or delete would put
// the old content back into retrieval through an unchanged sibling. Missing
// parents already fall back to eligible child hits; reparsing rebuilds them.
func (s *ChunkService) invalidateParentContexts(ctx context.Context, indexName, datasetID, documentID string, chunkIDs []string) error {
	parents := make(map[string]struct{})
	const batchSize = 512
	for start := 0; start < len(chunkIDs); start += batchSize {
		end := min(start+batchSize, len(chunkIDs))
		result, err := s.docEngine.Search(ctx, &types.SearchRequest{
			IndexNames: []string{indexName}, KbIDs: []string{datasetID},
			Limit: end - start, IncludeUnavailable: true,
			SelectFields: []string{"id", "mom_id"},
			Filter:       map[string]interface{}{"id": chunkIDs[start:end], "doc_id": documentID},
		})
		if err != nil {
			return fmt.Errorf("failed to find parent contexts: %w", err)
		}
		if result == nil {
			continue
		}
		for _, row := range result.Chunks {
			if id, _ := row["mom_id"].(string); id != "" {
				parents[id] = struct{}{}
			}
		}
	}
	if len(parents) == 0 {
		return nil
	}
	ids := make([]string, 0, len(parents))
	for id := range parents {
		ids = append(ids, id)
	}
	if _, err := s.docEngine.DeleteChunks(ctx, map[string]interface{}{
		"id": ids, "doc_id": documentID,
	}, indexName, datasetID); err != nil {
		return fmt.Errorf("failed to invalidate parent contexts: %w", err)
	}
	// Hidden parents are not counted as source chunks. Invalidation must not
	// change document counters; the subsequent source mutation owns those.
	return nil
}
