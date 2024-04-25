import { BaseLanguageModel } from "langchain/base_language";
import { Neo4jGraph } from "@langchain/community/graphs/neo4j_graph";
import { RunnablePassthrough } from "@langchain/core/runnables";
import initCypherGenerationChain from "./cypher-generation.chain";
import initCypherEvaluationChain from "./cypher-evaluation.chain";
import { saveHistory } from "../../history";
import { AgentToolInput } from "../../agent.types";
import { extractIds } from "../../../../utils";
import initGenerateAuthoritativeAnswerChain from "../../chains/authoritative-answer-generation.chain";

type CypherRetrievalThroughput = AgentToolInput & {
  context: string;
  output: string;
  cypher: string;
  results: Record<string, any> | Record<string, any>[];
  ids: string[];
};

/**
 * Use database the schema to generate and subsequently validate
 * a Cypher statement based on the user question
 *
 * @param {Neo4jGraph}        graph     The graph
 * @param {BaseLanguageModel} llm       An LLM to generate the Cypher
 * @param {string}            question  The rephrased question
 * @returns {string}
 */
async function recursivelyEvaluate(
  graph: Neo4jGraph,
  llm: BaseLanguageModel,
  question: string
): Promise<string> {
  const generationChain = await initCypherGenerationChain(graph, llm);
  const evaluatorChain = await initCypherEvaluationChain(llm);

  let cypher = await generationChain.invoke(question);

  let errors = ["N/A"];
  let tries = 0;

  while (tries < 5 && errors.length > 0) {
    tries++;

    try {
      // Evaluate Cypher
      const evaluation = await evaluatorChain.invoke({
        question,
        schema: graph.getSchema(),
        cypher,
        errors,
      });

      errors = evaluation.errors;
      cypher = evaluation.cypher;
    } catch (e: unknown) {}
  }

  // Bug fix: GPT-4 is adamant that it should use id() regardless of
  // the instructions in the prompt.  As a quick fix, replace it here
  cypher = cypher.replace(/\sid\(([^)]+)\)/g, " elementId($1)");

  return cypher;
}

/**
 * Attempt to get the results, and if there is a syntax error in the Cypher statement,
 * attempt to correct the errors.
 *
 * @param {Neo4jGraph}        graph  The graph instance to get the results from
 * @param {BaseLanguageModel} llm    The LLM to evaluate the Cypher statement if anything goes wrong
 * @param {string}            input  The input built up by the Cypher Retrieval Chain
 * @returns {Promise<Record<string, any>[]>}
 */
export async function getResults(
  graph: Neo4jGraph,
  llm: BaseLanguageModel,
  input: { question: string; cypher: string }
): Promise<any | undefined> {
  // Catch Cypher errors and pass to the Cypher evaluation chain
  let results;
  let retries = 0;
  let cypher = input.cypher;

  // Evaluation chain if an error is thrown by Neo4j
  const evaluationChain = await initCypherEvaluationChain(llm);

  while (results === undefined && retries < 5) {
    try {
      results = await graph.query(cypher);
      return results;
    } catch (e: any) {
      retries++;

      const evaluation = await evaluationChain.invoke({
        cypher,
        question: input.question,
        schema: graph.getSchema(),
        errors: [e.message],
      });

      cypher = evaluation.cypher;
    }
  }

  return results;
}

export default async function initCypherRetrievalChain(
  llm: BaseLanguageModel,
  graph: Neo4jGraph
) {
  const answerGeneration = await initGenerateAuthoritativeAnswerChain(llm);

  return (
    RunnablePassthrough
      // Generate and evaluate the Cypher statement
      .assign({
        cypher: (input: { rephrasedQuestion: string }) =>
          recursivelyEvaluate(graph, llm, input.rephrasedQuestion),
      })

      // Get results from database
      .assign({
        results: (input: { cypher: string; question: string }) =>
          getResults(graph, llm, input),
      })

      // Extract information
      .assign({
        // Extract _id fields
        ids: (input: Omit<CypherRetrievalThroughput, "ids">) =>
          extractIds(input.results),
        // Convert results to JSON output
        context: ({ results }: Omit<CypherRetrievalThroughput, "ids">) =>
          Array.isArray(results) && results.length == 1
            ? JSON.stringify(results[0])
            : JSON.stringify(results),
      })

      // Generate Output
      .assign({
        output: (input: CypherRetrievalThroughput) =>
          answerGeneration.invoke({
            question: input.rephrasedQuestion,
            context: input.context,
          }),
      })

      // Save response to database
      .assign({
        responseId: async (input: CypherRetrievalThroughput, options) => {
          saveHistory(
            options?.config.configurable.sessionId,
            "cypher",
            input.input,
            input.rephrasedQuestion,
            input.output,
            input.ids,
            input.cypher
          );
        },
      })
      // Return the output
      .pick("output")
  );
}
