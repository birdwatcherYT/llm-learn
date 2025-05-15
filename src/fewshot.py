from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")

from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

def escape_braces(text: str) -> str:
    """プロンプトで `{` や `}` をエスケープするために `{{` や `}}` に変換する"""
    return text.replace("{", "{{").replace("}", "}}")

examples = [
    {
        "num": 1,
        "input": """[
 {"role": "user", "content": "最近読書を始めたよ"},
 {"role": "assistant", "content": "いいですね！どんな本を読んでいるんですか？"},
 {"role": "user", "content": "ミステリー小説を読んでるよ。すごく面白い！"},
 {"role": "assistant", "content": "ミステリーは展開が気になりますよね。どんなストーリーなんですか？"},
 {"role": "user", "content": "探偵が事件を解決していく話なんだけど、伏線がすごくて驚かされる！"},
 {"role": "assistant", "content": "伏線が巧妙なミステリーって最高ですね！最後まで楽しめそうですね。"},
 {"role": "user", "content": "そうなんだよ！続きが気になってどんどん読んじゃう。"}
]
""",
        "output": "userは最近読書を始め、ミステリー小説を読んでいることを述べている。探偵が事件を解決する話で、伏線が巧妙で驚かされると感じており、続きが気になってどんどん読んでしまうほど楽しんでいることを伝えている。",
    },
    {
        "num": 2,
        "input": """[
 {"role": "user", "content": "料理に挑戦してみたよ"},
 {"role": "assistant", "content": "いいですね！何を作ったんですか？"},
 {"role": "user", "content": "オムライスを作ってみた！思ったよりうまくできたよ。"},
 {"role": "assistant", "content": "オムライス、美味しそうですね！うまくできたならすごいです！"},
 {"role": "user", "content": "卵をふわふわにするのが難しかったけど、頑張ったよ。"},
 {"role": "assistant", "content": "ふわふわな卵、憧れますね！コツとか掴めましたか？"},
 {"role": "user", "content": "火加減が大事だって分かったよ。次はもっと上手に作りたいな！"}
]
""",
        "output": "userは料理に挑戦し、オムライスを作ったことを述べている。卵をふわふわにするのが難しかったが、頑張って作り、火加減が重要であることに気づいたと話している。次はさらに上手に作りたいと考えていることを伝えている。",
    },
]


example_prompt = PromptTemplate.from_template(
    "## 入出力例{num}\n### 入力\n```json\n{input}\n```\n### 出力\n{output}"
)
fewshot_prompt = FewShotPromptTemplate(
    examples=[
        {
            "num": example["num"],
            "input": escape_braces(example["input"]),
            "output": example["output"],
        }
        for example in examples
    ],
    example_prompt=example_prompt,
    suffix="\n",
    input_variables=[],
)
print("-"*50)
print(fewshot_prompt.format())
print("-"*50)


prompt = """
与えられた会話履歴をなるべく情報を落とすことなく要約してください

# examples
{few_shot_examples}

# 会話履歴
```json
{chat_history}
```
"""

prompt_template = PromptTemplate.from_template(
    template=prompt,
    partial_variables={"few_shot_examples": fewshot_prompt.format()},
)

chain = prompt_template | llm | StrOutputParser()
result = chain.invoke("""
 {"role": "user", "content": "最近ジムに通い始めたよ"},
 {"role": "assistant", "content": "すごいですね！どんなトレーニングをしているんですか？"},
 {"role": "user", "content": "筋トレとランニングをしてるよ。"},
 {"role": "assistant", "content": "バランスよく鍛えてるんですね！筋トレはどんな種目をやってますか？"},
 {"role": "user", "content": "スクワットとかベンチプレスをしてるよ。"},
 {"role": "assistant", "content": "しっかり鍛えてますね！筋肉痛とか大丈夫ですか？"},
 {"role": "user", "content": "最初は結構痛かったけど、少しずつ慣れてきたよ。"}
""")
print(result)
