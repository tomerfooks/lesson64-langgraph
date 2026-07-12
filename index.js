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

import { tool } from "@langchain/core/tools";
import { z } from "zod";
import dotenv from "dotenv";
dotenv.config();

// 1) המודל. שנו את משתני הסביבה כדי לעבוד עם ספק אחר.



const sendTelegramMessage = async ({ message }) => {
  const token = process.env.TELEGRAM_BOT_TOKEN;
  const chatId = process.env.TELEGRAM_CHAT_ID;
  if (!token || !chatId)
    return "Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing in .env";

  const data = await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id: chatId, text: message }),
  });

  const result = data.json();
  if (!result.ok) return `Error sending message: ${result.description}`;
  return "Message sent successfully";
};

// כלי: שולח הודעה לצ'אט בטלגרם דרך ה-Bot API (טוקן ומזהה צ'אט מ-.env).
const sendTelegramMessageTool = tool(sendTelegramMessage, {
  name: "send_telegram_message",
  description: "שולח הודעת טקסט למשתמש בטלגרם",
  schema: z.object({ message: z.string() }),
});

const getTelegramMessages = async () => {
  const token = process.env.TELEGRAM_BOT_TOKEN;
  if (!token) return "Error: TELEGRAM_BOT_TOKEN missing in .env";
  const data = await fetch(`https://api.telegram.org/bot${token}/getUpdates`);
  const result = await data.json();

  // מסננים רק עדכונים שמכילים הודעת טקסט ולוקחים את האחרונות
  const messages = result.result
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
};

// כלי: קורא את ההודעות האחרונות שנשלחו לבוט בטלגרם דרך getUpdates.
const getTelegramMessagesTool = tool(getTelegramMessages, {
  name: "get_telegram_messages",
  description: "קורא את ההודעות האחרונות שנשלחו לבוט בטלגרם",
  schema: z.object({
    limit: z
      .number()
      .optional()
      .describe("כמה הודעות אחרונות להחזיר (ברירת מחדל 5)"),
  }),
});


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
  searchProductsTool,
  sendTelegramMessageTool,
  getTelegramMessagesTool,
];
const modelWithTools = model.bindTools(tools);

// 3) צומת "agent": שולח את ההודעות למודל ומחזיר את תשובתו.
async function callModel(state) {
  const answer = await modelWithTools.invoke(state.messages);
  return { messages: [answer] };
}

function shouldContinue(state) {
  let lastMessage = state.messages[state.messages.length - 1];
  if (lastMessage.tool_calls?.length) 
    return "tools";

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
