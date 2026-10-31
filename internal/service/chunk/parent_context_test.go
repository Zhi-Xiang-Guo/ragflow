package chunk

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"ragflow/internal/dao"
	"ragflow/internal/engine"
	"ragflow/internal/engine/types"
	"ragflow/internal/service"
	"ragflow/internal/service/nlp"
)

// A small index model exercises the service mutation followed by the real
// parent-expansion consumer. Deployed API/model-backed evidence is separate.
type parentContextEngine struct {
	engine.DocEngine
	rows    map[string]map[string]interface{}
	fail    string
	updates int
}

func parentContextMatches(row map[string]interface{}, filter map[string]interface{}) bool {
	for key, expected := range filter {
		match := false
		switch values := expected.(type) {
		case []string:
			for _, value := range values {
				match = match || row[key] == value
			}
		case []interface{}:
			for _, value := range values {
				match = match || row[key] == value
			}
		default:
			match = row[key] == expected
		}
		if !match {
			return false
		}
	}
	return true
}

func (e *parentContextEngine) Search(_ context.Context, req *types.SearchRequest) (*types.SearchResult, error) {
	if e.fail == "lookup" {
		return nil, fmt.Errorf("parent lookup unavailable")
	}
	result := &types.SearchResult{}
	for _, row := range e.rows {
		if parentContextMatches(row, req.Filter) {
			result.Chunks = append(result.Chunks, copyMap(row))
		}
	}
	return result, nil
}

func (e *parentContextEngine) GetChunk(_ context.Context, _, id string, _ []string) (interface{}, error) {
	if row, ok := e.rows[id]; ok {
		return copyMap(row), nil
	}
	return nil, fmt.Errorf("missing chunk %s", id)
}

func (e *parentContextEngine) UpdateChunks(_ context.Context, filter, values map[string]interface{}, _, _ string) error {
	e.updates++
	for _, row := range e.rows {
		if parentContextMatches(row, filter) {
			for k, v := range values {
				row[k] = v
			}
		}
	}
	return nil
}

func (e *parentContextEngine) DeleteChunks(_ context.Context, filter map[string]interface{}, _, _ string) (int64, error) {
	if e.fail == "delete" {
		return 0, fmt.Errorf("parent deletion unavailable")
	}
	var count int64
	for id, row := range e.rows {
		if parentContextMatches(row, filter) {
			delete(e.rows, id)
			count++
		}
	}
	return count, nil
}

func TestChildMutationCannotReturnStaleParentContext(t *testing.T) {
	for _, operation := range []string{"disable-batch", "disable-single", "edit", "delete"} {
		t.Run(operation, func(t *testing.T) {
			db := setupChunkTestDB(t)
			pushChunkTestDB(t, db)
			insertChunkTestUserTenant(t, "user-1", "tenant-1")
			insertChunkTestKB(t, "kb-1", "tenant-1")
			insertChunkTestDoc(t, "doc-1", "kb-1")
			e := &parentContextEngine{rows: map[string]map[string]interface{}{
				"child-a":        {"id": "child-a", "doc_id": "doc-1", "mom_id": "parent-1", "content_with_weight": "The obsolete access code is OLD-123.", "available_int": 1},
				"child-b":        {"id": "child-b", "doc_id": "doc-1", "mom_id": "parent-1", "content_with_weight": "The warranty lasts twelve months.", "available_int": 1},
				"parent-1":       {"id": "parent-1", "doc_id": "doc-1", "kb_id": "kb-1", "content_with_weight": "The obsolete access code is OLD-123. The warranty lasts twelve months.", "available_int": 0},
				"foreign-parent": {"id": "foreign-parent", "doc_id": "doc-other", "content_with_weight": "Unrelated context"},
			}}
			svc := &ChunkService{docEngine: e, kbDAO: dao.NewKnowledgebaseDAO(),
				decrementChunkStatsFunc: func(string, string, int64, int64, float64) error { return nil },
				markWikiDirtyFunc:       func(string, string, string, []string) {},
			}
			ctx := chunkTestContext(t, "user-1", "tenant-1")
			var err error
			switch operation {
			case "disable-batch":
				err = svc.SwitchChunks(ctx, "user-1", "kb-1", "doc-1", 0, []string{"child-a"})
			case "disable-single":
				available := false
				err = svc.UpdateChunk(ctx, &service.UpdateChunkRequest{DatasetID: "kb-1", DocumentID: "doc-1", ChunkID: "child-a", Available: &available}, "user-1")
			case "edit":
				content := "The current access code is NEW-456."
				err = svc.UpdateChunk(ctx, &service.UpdateChunkRequest{DatasetID: "kb-1", DocumentID: "doc-1", ChunkID: "child-a", Content: &content}, "user-1")
			case "delete":
				_, err = svc.RemoveChunks(ctx, &service.RemoveChunksRequest{DatasetID: "kb-1", DocID: "doc-1", ChunkIDs: []string{"child-a"}}, "user-1")
			}
			if err != nil {
				t.Fatal(err)
			}
			hit := copyMap(e.rows["child-b"])
			hit["chunk_id"], hit["kb_id"] = "child-b", "kb-1"
			chunks := nlp.RetrievalByChildren([]map[string]interface{}{hit}, []string{"tenant-1"}, e, ctx)
			if len(chunks) != 1 {
				t.Fatalf("remaining sibling was lost: %#v", chunks)
			}
			text, _ := chunks[0]["content_with_weight"].(string)
			if strings.Contains(text, "OLD-123") || !strings.Contains(text, "twelve months") {
				t.Fatalf("retrieval after %s returned stale context: %q", operation, text)
			}
			if e.rows["foreign-parent"] == nil {
				t.Fatal("unrelated parent was invalidated")
			}
		})
	}
}

func TestParentInvalidationFailureDoesNotCommitSourceMutation(t *testing.T) {
	for _, failure := range []string{"lookup", "delete"} {
		t.Run(failure, func(t *testing.T) {
			db := setupChunkTestDB(t)
			pushChunkTestDB(t, db)
			insertChunkTestUserTenant(t, "user-1", "tenant-1")
			insertChunkTestKB(t, "kb-1", "tenant-1")
			insertChunkTestDoc(t, "doc-1", "kb-1")
			e := &parentContextEngine{fail: failure, rows: map[string]map[string]interface{}{
				"child":  {"id": "child", "doc_id": "doc-1", "mom_id": "parent", "available_int": 1},
				"parent": {"id": "parent", "doc_id": "doc-1", "available_int": 0},
			}}
			svc := &ChunkService{docEngine: e, kbDAO: dao.NewKnowledgebaseDAO(), markWikiDirtyFunc: func(string, string, string, []string) {}}
			err := svc.SwitchChunks(chunkTestContext(t, "user-1", "tenant-1"), "user-1", "kb-1", "doc-1", 0, []string{"child"})
			if err == nil || e.updates != 0 || e.rows["child"]["available_int"] != 1 {
				t.Fatalf("source changed despite failed parent invalidation: err=%v updates=%d", err, e.updates)
			}
		})
	}
}
