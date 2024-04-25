import { BaseChatModel } from "langchain/chat_models/base";
import { Embeddings } from "@langchain/core/embeddings";
import { Neo4jGraph } from "@langchain/community/graphs/neo4j_graph";
import initCypherRetrievalChain from "./cypher/cypher-retrieval.chain";
import initVectorRetrievalChain from "./vector-retrieval.chain";
import { DynamicStructuredTool } from "@langchain/community/tools/dynamic";
import { AgentToolInputSchema } from "../agent.types";
import { RunnableConfig } from "langchain/runnables";

export default async function initTools(
  llm: BaseChatModel,
  embeddings: Embeddings,
  graph: Neo4jGraph
): Promise<DynamicStructuredTool[]> {
  // Initiate chains
  const cypherChain = await initCypherRetrievalChain(llm, graph);
  const retrievalChain = await initVectorRetrievalChain(llm, embeddings);

  return [
    new DynamicStructuredTool({
      name: "graph-cypher-retrieval-chain",
      description:
        "For retrieving Product information from the database including product recommendations and description",
      schema: AgentToolInputSchema,
      func: (input, _runManager, config) => cypherChain.invoke(input, config),
    }),
    new DynamicStructuredTool({
      name: "graph-vector-retrieval-chain",
      description:
        "For finding products, comparing products by their description or recommending a product",
      schema: AgentToolInputSchema,
      func: (input, _runManager: any, config) =>
        retrievalChain.invoke(input, config),
    }),
  ];
}
