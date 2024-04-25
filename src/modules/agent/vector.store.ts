import { EmbeddingsInterface } from "@langchain/core/embeddings";
import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";

/**
 * Create a new vector search index that uses the existing index.
 *
 * @param {EmbeddingsInterface} embeddings  The embeddings model
 * @returns {Promise<Neo4jVectorStore>}
 */
export default async function initVectorStore(
  embeddings: EmbeddingsInterface
): Promise<Neo4jVectorStore> {
  const vectorStore = await Neo4jVectorStore.fromExistingIndex(embeddings, {
    url: process.env.NEO4J_URI as string,
    username: process.env.NEO4J_USERNAME as string,
    password: process.env.NEO4J_PASSWORD as string,
    indexName: "description",
    textNodeProperty: "value",
    embeddingNodeProperty: "embedding",
    retrievalQuery: `
      RETURN
        node.value AS text,
        score,
        { _id: elementid(node) } AS metadata
    `,
  });
  return vectorStore;
}
