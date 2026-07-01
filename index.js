
import { StateGraph, MessagesAnnotation, START, END } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
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
      `https://dummyjson.com/products/search?q=${encodeURIComponent(query)}`
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
  }
);

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
const question = process.argv.slice(2).join(" ") || "חפש לי טלפון";

const result = await app.invoke({
  messages: [{ role: "user", content: question }],
});

// ההודעה האחרונה היא התשובה הסופית של הסוכן
const answer = result.messages[result.messages.length - 1];
console.log(answer.content);