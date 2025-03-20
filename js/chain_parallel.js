import { config } from "dotenv";
config();

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
const llm = new ChatGoogleGenerativeAI({ model: "gemini-1.5-flash" });

import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableLambda, RunnableParallel } from "@langchain/core/runnables";


const prompt1 = PromptTemplate.fromTemplate(
    "次の文章から趣味を抽出して1つの単語で回答してください。\n{user_input}"
);
const chain1 = prompt1.pipe(llm).pipe(new StringOutputParser());

const prompt2 = PromptTemplate.fromTemplate(
    "次の文章から職業を抽出して1つの単語で回答してください。\n{user_input}"
);
const chain2 = prompt2.pipe(llm).pipe(new StringOutputParser());

const prompt3 = PromptTemplate.fromTemplate(
    "次の記述を{language}に翻訳してください。\n{hobby}, {occupation}"
);
const chain3 = prompt3.pipe(llm).pipe(new StringOutputParser());

const chain = RunnableParallel.from({
    hobby: chain1,
    occupation: chain2,
    language: RunnableLambda.from((x) => "英語"),
}).pipe(chain3);

const result = await chain.invoke(
    { user_input: "普段はAIエンジニアをしていますが、休日には外に出て鳥を観察しに行き、撮影をします" },
);

console.log(result);
