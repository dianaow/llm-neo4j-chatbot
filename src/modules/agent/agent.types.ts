import { z } from "zod";

export interface ChatAgentInput {
  input: string;
}

export interface AgentToolInput {
  input: string;
  rephrasedQuestion: string;
}

export const AgentToolInputSchema = z.object({
  input: z.string().describe("The original input sent by the user"),
  rephrasedQuestion: z
    .string()
    .describe(
      "A rephrased version of the original question based on the conversation history"
    ),
});
