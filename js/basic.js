import { config } from "dotenv";
config();

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

const llm = new ChatGoogleGenerativeAI({ model: "gemini-1.5-flash" });

const result = await llm.invoke("LLMについて教えて下さい");

console.log(result.content);
