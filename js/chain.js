import { config } from "dotenv";
config();

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
const llm = new ChatGoogleGenerativeAI({ modelName: "gemini-1.5-flash" });

import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";

const prompt1 = PromptTemplate.fromTemplate(
    "次の文章から趣味を抽出して1つの単語で回答してください。\n{user_input}",
);
const chain1 = prompt1.pipe(llm).pipe(new StringOutputParser()).pipe(hobby => { return { hobby: hobby } });;

const prompt2 = PromptTemplate.fromTemplate(
    "次の記述を英語に翻訳してください。\n{hobby}",
);
const chain2 = prompt2.pipe(llm).pipe(new StringOutputParser());

const chain = chain1.pipe(chain2);

const result = await chain.invoke(
    { user_input: "普段はAIエンジニアをしていますが、休日には外に出て鳥を観察しに行き、撮影をします" }
);

console.log(result);
