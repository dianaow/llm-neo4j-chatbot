import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { BaseLanguageModel } from "langchain/base_language";

export type GenerateAuthoritativeAnswerInput = {
  question: string;
  context: string | undefined;
};

export default function initGenerateAuthoritativeAnswerChain(
  llm: BaseLanguageModel
): RunnableSequence<GenerateAuthoritativeAnswerInput, string> {

  const answerQuestionPrompt = PromptTemplate.fromTemplate(`
    Use the following context to answer the following question.
    The context is provided by an authoritative source, you must never doubt
    it or attempt to use your pre-trained knowledge to correct the answer.

    Make the answer sound like it is a response to the question.
    Do not mention that you have based your response on the context.

    Here is an example:

    Question: Who played Woody in Toy Story?
    Context: ['role': 'Woody', 'actor': 'Tom Hanks']
    Response: Tom Hanks played Woody in Toy Story.

    If no context is provided, say that you don't know,
    don't try to make up an answer, do not fall back to your internal knowledge.
    If no context is provided you may also ask for clarification.

    Include links and sources where possible.

    Question:
    {question}

    Context:
    {context}
  `);

  return RunnableSequence.from<GenerateAuthoritativeAnswerInput, string>([
    RunnablePassthrough.assign({
      context: ({ context }) =>
        context == undefined || context === "" ? "I don't know" : context,
    }),
    answerQuestionPrompt,
    llm,
    new StringOutputParser(),
  ]);
}