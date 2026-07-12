import dotenv from "dotenv";
dotenv.config();

import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({
  model: process.env.LLM_MODEL || "meta/llama-3.3-70b-instruct",
  apiKey: process.env.LLM_API_KEY || "no-key",
  configuration: {
    baseURL: process.env.LLM_BASE_URL || "https://integrate.api.nvidia.com/v1",
  },
});

export default model