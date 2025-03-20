import { config } from "dotenv";
config();

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
const llm = new ChatGoogleGenerativeAI({ model: "gemini-1.5-flash" });

import { PromptTemplate } from "@langchain/core/prompts";
import { StructuredOutputParser } from "langchain/output_parsers";
import { z } from "zod";


const PROMPT = `
「入力文」から「趣味」「職業」「年齢」を抽出してください。
該当する項目がなければnullで埋めてください。

# 入力文
{input_str}

# Format instructions
{format_instructions}
`;

const responseSchemas = z.object({
    "趣味": z.string().describe("ユーザーの入力文から判断できる趣味"),
    "職業": z.string().describe("ユーザーの入力文から判断できる職業"),
    "年齢": z.number().describe("ユーザーの入力文から判断できる年齢"),
});
const outputParser = StructuredOutputParser.fromZodSchema(responseSchemas);

const promptTemplate = new PromptTemplate({
    template: PROMPT,
    inputVariables: ["input_str"],
    partialVariables: { format_instructions: outputParser.getFormatInstructions() },
});

const chain = promptTemplate.pipe(llm).pipe(outputParser);
const result = await chain.invoke({ input_str: "Webデザイナーをしています。山を登ることが趣味です。生まれてから20年経過しました。" });
console.log(result); // -> { 趣味: '山登り', 職業: 'Webデザイナー', 年齢: 20 }
console.log(typeof result); // -> object
