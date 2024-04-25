import { BaseLanguageModel } from "langchain/base_language";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Neo4jGraph } from "@langchain/community/graphs/neo4j_graph";

export default async function initCypherGenerationChain(
  graph: Neo4jGraph,
  llm: BaseLanguageModel
) {
  const cypherPrompt = PromptTemplate.fromTemplate(`
    You are a Neo4j Developer translating user questions into Cypher to answer questions
    about skincare products and provide recommendations.
    Convert the user's question into a Cypher statement based on the schema.

    You must:
    * Only use the nodes, relationships and properties mentioned in the schema.
    * When required, \`IS NOT NULL\` to check for property existence, and not the exists() function.
    * Use the \`elementId()\` function to return the unique identifier for a node or relationship as \`_id\`.
      For example:
      \`\`\`
      MATCH (p:Product)-[:hasBenefits]->(b:benefits)
      WHERE b.value CONTAINS 'brightening'
      RETURN p.title AS Product, elementId(m) AS _id
      \`\`\`
    * Limit the maximum number of results to 10.
    * Respond with only a Cypher statement.  No preamble.

    Example Question: Recommend a product with brightening properties?
    Example Cypher:
    MATCH (p:Product)-[:hasBenefits]->(b:benefits)
    WHERE b.value CONTAINS 'brightening'
    RETURN p.title AS Product, elementId(m) AS _id

    Schema:
    {schema}

    Question:
    {question}
  `);

  return RunnableSequence.from<string, string>([
    {
      question: new RunnablePassthrough(),
      schema: () => graph.getSchema(),
    },
    cypherPrompt,
    llm,
    new StringOutputParser(),
  ]);
    
}
