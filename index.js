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
      .map((product) => `${product.title} - ${product.price}$`)
      .join("\n");
  },
  {
    name: "search_products",
    description: "מחפש מוצרים בחנות לפי מילת חיפוש ומחזיר שם ומחיר",
    schema: z.object({ query: z.string() }),
  },
);

// כלי: שולח הודעה לצ'אט בטלגרם דרך ה-Bot API (טוקן ומזהה צ'אט מ-.env).
const sendTelegramMessage = tool(
  async ({ message }) => {
    const token = process.env.TELEGRAM_BOT_TOKEN;
    const chatId = process.env.TELEGRAM_CHAT_ID;
    if (!token || !chatId)
      return "Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing in .env";

    const data = await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chat_id: chatId, text: message }),
    }).then((res) => res.json());

    if (!data.ok) return `Error sending message: ${data.description}`;
    return "Message sent successfully";
  },
  {
    name: "send_telegram_message",
    description: "שולח הודעת טקסט למשתמש בטלגרם",
    schema: z.object({ message: z.string() }),
  },
);

// כלי: קורא את ההודעות האחרונות שנשלחו לבוט בטלגרם דרך getUpdates.
const getTelegramMessages = tool(
  async ({ limit }) => {
    const token = process.env.TELEGRAM_BOT_TOKEN;
    if (!token) return "Error: TELEGRAM_BOT_TOKEN missing in .env";

    const data = await fetch(
      `https://api.telegram.org/bot${token}/getUpdates`,
    ).then((res) => res.json());

    if (!data.ok) return `Error reading messages: ${data.description}`;

    // מסננים רק עדכונים שמכילים הודעת טקסט ולוקחים את האחרונות
    const messages = data.result
      .filter((update) => update.message?.text)
      .slice(-(limit ?? 5))
      .map((update) => {
        const msg = update.message;
        const from = msg.from?.username || msg.from?.first_name || "unknown";
        const date = new Date(msg.date * 1000).toISOString();
        return `[${date}] ${from}: ${msg.text}`;
      });

    if (!messages.length) return "No new messages";
    return messages.join("\n");
  },
  {
    name: "get_telegram_messages",
    description: "קורא את ההודעות האחרונות שנשלחו לבוט בטלגרם",
    schema: z.object({
      limit: z.number().optional().describe("כמה הודעות אחרונות להחזיר (ברירת מחדל 5)"),
    }),
  },
);

// כלי: שולח הודעת וואטסאפ דרך Green API (מזהה מופע וטוקן מ-.env).
const sendWhatsappMessage = tool(
  async ({ message, phone }) => {
    const idInstance = process.env.GREEN_API_ID_INSTANCE;
    const apiToken = process.env.GREEN_API_TOKEN;
    const apiUrl = process.env.GREEN_API_URL || "https://api.green-api.com";
    const targetPhone = phone || process.env.WHATSAPP_PHONE;

    if (!idInstance || !apiToken)
      return "Error: GREEN_API_ID_INSTANCE or GREEN_API_TOKEN missing in .env";
    if (!targetPhone)
      return "Error: no phone number given and WHATSAPP_PHONE missing in .env";

    // Green API מצפה למספר בפורמט בינלאומי בלי + ועם סיומת @c.us
    const chatId = `${targetPhone.replace(/\D/g, "")}@c.us`;

    const res = await fetch(
      `${apiUrl}/waInstance${idInstance}/sendMessage/${apiToken}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chatId, message }),
      },
    );

    // Green API מחזיר HTML (לא JSON) כשהמזהה או הטוקן שגויים
    const text = await res.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      return `Error sending message: HTTP ${res.status} - ${text.slice(0, 200)}`;
    }

    if (!data.idMessage) return `Error sending message: ${JSON.stringify(data)}`;
    return "Message sent successfully";
  },
  {
    name: "send_whatsapp_message",
    description: "שולח הודעת וואטסאפ למספר טלפון דרך Green API",
    schema: z.object({
      message: z.string(),
      phone: z
        .string()
        .optional()
        .describe("מספר טלפון בפורמט בינלאומי, למשל 972501234567 (ברירת מחדל מ-.env)"),
    }),
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

const tools = [
  searchProducts,
  sendTelegramMessage,
  getTelegramMessages,
  sendWhatsappMessage,
];
const modelWithTools = model.bindTools(tools);

// 3) צומת "agent": שולח את ההודעות למודל ומחזיר את תשובתו.
async function callModel(state) {
  const answer = await modelWithTools.invoke(state.messages);
  return { messages: [answer] };
}

// 4) החלטה: אם המודל ביקש כלי -> נלך לצומת "tools". אחרת -> סוף.
function shouldContinue(state) {
  let lastMessage = state.messages[state.messages.length - 1];
  if (lastMessage.tool_calls?.length) return "tools";



  const forbiddenWords = ["Mascara", "Makeup"];
  if (forbiddenWords.some((word) => lastMessage.content?.includes(word))) {
    console.error("forbidden word detected in the answer, stopping the flow.");
    return END;
  }

  return END;
}

// 1: 

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
