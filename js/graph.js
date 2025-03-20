import { config } from "dotenv";
config();

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { StateGraph } from "@langchain/langgraph";
const chain = new ChatGoogleGenerativeAI({ model: "gemini-1.5-flash" }).pipe(new StringOutputParser())


// nodeに使う関数定義（returnした値が更新されます）
async function getName(state, config) {
    return { name: await chain.invoke(`${state.names}以外の鳥の名前を1つあげて`) };
}

async function getFeature(state, config) {
    return { feature: await chain.invoke(`${state.name}の特徴を一言で`) };
}

async function getArea(state, config) {
    return { area: await chain.invoke(`${state.name}の生息地を一言で`) };
}

async function output(state, config) {
    console.log(state);
    return { names: [...state.names, state.name] };
}

async function finish(state, config) {
    return {};
}

// 分岐
function judge(state, config) {
    return state.names.length < 3 ? "name_node" : "finish";
}

// ワークフロー定義
const workflow = new StateGraph({
    channels: {
        name: { initial: "" },
        area: { initial: "" },
        feature: { initial: "" },
        names: { initial: [] },
    },
});

// node追加
workflow.addNode("name_node", getName);
workflow.addNode("feature_node", getFeature);
workflow.addNode("area_node", getArea);
workflow.addNode("output", output);
workflow.addNode("finish", finish);

// edge追加
workflow.addEdge("name_node", "area_node");
workflow.addEdge("name_node", "feature_node");
workflow.addEdge("area_node", "output");
workflow.addEdge("feature_node", "output");

// 始点
workflow.addEdge("__start__", "name_node");

// 終点
workflow.addEdge("finish", "__end__");

// 分岐
workflow.addConditionalEdges("output", judge);

// コンパイル
const graph = workflow.compile();

const result = await graph.invoke({ names: [] });
console.log("最終アウトプット:", result);
console.log((await graph.getGraphAsync()).drawMermaid());
