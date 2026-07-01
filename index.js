import {
  StateGraph,
  MessagesAnnotation,
  START,
  END,
} from "@langchain/langgraph";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { ToolNode } from "@langchain/langgraph/prebuilt";

import {
  mapChatMessagesToStoredMessages,
  mapStoredMessagesToChatMessages,
} from "@langchain/core/messages";

import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import dotenv from "dotenv";

dotenv.config();

// 1) המודל. שנו את משתני הסביבה כדי לעבוד עם ספק אחר.
const model = new ChatOpenAI({
  model: process.env.LLM_MODEL || "meta/llama-3.3-70b-instruct",
  apiKey: process.env.LLM_API_KEY || "no-key",
  configuration: {
    baseURL: process.env.LLM_BASE_URL || "https://integrate.api.nvidia.com/v1",
  },
});

// 2) כלי (Tool): מחפש מוצרים לפי מילת חיפוש, דרך DummyJSON (בלי מפתח API).
const searchProducts = tool(
  async ({ query }) => {
    const data = await fetch(
      `https://dummyjson.com/products/search?q=${encodeURIComponent(query)}`,
    ).then((res) => res.json());
    // לוקחים עד 5 תוצאות ומחזירים שם ומחיר של כל מוצר
    return data.products
      .slice(0, 5)
      .map((product) => `${product.title} - ${product.price}$`)
      .join("\n");
  },
  {
    name: "search_products",
    description: "מחפש מוצרים בחנות לפי מילת חיפוש ומחזיר שם ומחיר",
    schema: z.object({ query: z.string() }),
  },
);

const MEMORY_FILE = "memory.json";
const SYSTEM_PROMPT_FILE = "system-prompt.json";

// טוען היסטוריית הודעות מהקובץ (אם קיים) וממיר לאובייקטי הודעה של LangChain.
function loadMemory() {
  if (!existsSync(MEMORY_FILE))
    return [
      { role: "system", content: readFileSync(SYSTEM_PROMPT_FILE, "utf8") },
    ];
  const stored = JSON.parse(readFileSync(MEMORY_FILE, "utf8"));
  return mapStoredMessagesToChatMessages(stored);
}

// שומר את כל ההיסטוריה לקובץ בפורמט שניתן לשחזר בהמשך (כולל קריאות לכלים).
function saveMemory(messages) {
  const stored = mapChatMessagesToStoredMessages(messages);
  writeFileSync(MEMORY_FILE, JSON.stringify(stored, null, 2), "utf8");
}

const tools = [searchProducts];
const modelWithTools = model.bindTools(tools);

// 3) צומת "agent": שולח את ההודעות למודל ומחזיר את תשובתו.
async function callModel(state) {
  const answer = await modelWithTools.invoke(state.messages);
  return { messages: [answer] };
}

// 4) החלטה: אם המודל ביקש כלי -> נלך לצומת "tools". אחרת -> סוף.
function shouldContinue(state) {
  const lastMessage = state.messages[state.messages.length - 1];
  if (lastMessage.tool_calls?.length) {
    return "tools";
  }
  return END;
}

// 5) בונים את הגרף: agent -> (כלי?) -> agent -> ... -> סוף
const app = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", new ToolNode(tools))
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue, { tools: "tools", [END]: END })
  .addEdge("tools", "agent")
  .compile();

// 6) הרצה
const question = process.argv.slice(2).join(" ");

const memory = loadMemory();

const result = await app.invoke({
  messages: [...memory, { role: "user", content: question }],
});

// ההודעה האחרונה היא התשובה הסופית של הסוכן
const answer = result.messages[result.messages.length - 1];

saveMemory(result.messages);

console.log(answer.content);
