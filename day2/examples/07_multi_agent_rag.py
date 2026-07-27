"""多角色 RAG 範例：把回答、轉譯與審查拆成三個明確步驟。"""

import argparse
from typing import NotRequired, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from examples.knowledge_base import format_documents, open_local_vector_store
from examples.model_factory import create_chat_model, create_embeddings
from examples.rag_pipeline import generate_answer, validate_citations
from examples.schemas import EvidenceReview, RagAnswer

# 教學轉譯 Agent 只接收草稿文字，不直接讀取原始文件。
# 它的工作是依 audience 調整說法，例如改寫成適合家長或學生閱讀的文字。
# Prompt 特別限制它不得增加新規定，避免「改寫」變成「自行補充」。
ADAPT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是教學轉譯 Agent。依指定讀者重寫答案，保留原有事實與限制，"
            "不得加入草稿或來源中沒有的規定。只輸出重寫後的答案文字。",
        ),
        ("human", "讀者：{audience}\n原始草稿：{draft}"),
    ]
)

# 證據審查 Agent 會同時看到候選答案與檢索文件，判斷每項主張是否有依據。
# with_structured_output(EvidenceReview) 會要求模型依固定欄位回傳：
# grounded、issues、revised_answer，方便後續 Python 程式判斷，而不是解析自由文字。
REVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是證據審查 Agent。逐項檢查候選答案是否受到檢索證據支持。"
            "若有未支持的主張，列入 issues 並在 revised_answer 移除或改成需人工確認。"
            "不得利用模型記憶補充規定。",
        ),
        (
            "human",
            "問題：{question}\n\n候選答案：{candidate}\n\n檢索證據：\n{context}",
        ),
    ]
)


# LangGraph 的 State 是所有節點共同傳遞的資料。
# question、audience、trace 在流程開始時就有值；其餘欄位由各節點依序加入，
# 因此使用 NotRequired 表示「初始 State 可以暫時沒有這個欄位」。
class MultiAgentState(TypedDict):
    # 使用者輸入
    question: str
    audience: str

    # retrieval_node 寫入檢索到的文件
    documents: NotRequired[list[Document]]

    # drafting_agent 寫入含有回答、引用與證據狀態的結構化草稿
    draft: NotRequired[RagAnswer]

    # teaching_agent 寫入依讀者改寫後的純文字
    candidate: NotRequired[str]

    # evidence_agent 寫入結構化審查結果
    review: NotRequired[EvidenceReview]

    # finalize_node 寫入最後要輸出的答案
    final_answer: NotRequired[RagAnswer]

    # 每個節點都會追加一筆文字，方便觀察實際執行順序
    trace: list[str]


def build_multi_agent_graph(model: BaseChatModel, vector_store: object):
    # 把 Prompt、Chat Model、輸出解析器串成一條 Chain。
    adapt_chain = ADAPT_PROMPT | model | StrOutputParser()

    # Chat Model 指定不同輸出格式；這裡要求回傳 EvidenceReview。
    review_model = model.with_structured_output(EvidenceReview)

    def retrieval_node(state: MultiAgentState) -> dict:
        # 這是固定 RAG：每一題都直接查詢 Qdrant，不讓模型決定是否檢索。
        documents = vector_store.similarity_search(state["question"], k=4)
        return {
            "documents": documents,
            "trace": state["trace"] + ["檢索節點：取得校務文件段落"],
        }

    def drafting_agent(state: MultiAgentState) -> dict:
        # generate_answer 會把問題和文件交給模型，並要求回傳 RagAnswer，
        # 所以草稿除了答案文字，也包含 citations 和 insufficient_evidence。
        draft = generate_answer(model, state["question"], state["documents"])
        return {
            "draft": draft,
            "trace": state["trace"] + ["回答 Agent：依證據產生結構化草稿"],
        }

    def teaching_agent(state: MultiAgentState) -> dict:
        # 這裡只改寫答案文字；原本的 citations 仍保存在 draft，沒有交給此 Agent 修改。
        candidate = adapt_chain.invoke(
            {"audience": state["audience"], "draft": state["draft"].answer}
        ).strip()
        return {
            "candidate": candidate,
            "trace": state["trace"] + ["教學轉譯 Agent：調整讀者語氣"],
        }

    def evidence_agent(state: MultiAgentState) -> dict:
        # format_documents 會把 Document 轉成含 source、chunk_id 和內文的字串，
        # 讓審查 Agent 可以逐項比對候選答案與原始證據。
        prompt_value = REVIEW_PROMPT.invoke(
            {
                "question": state["question"],
                "candidate": state["candidate"],
                "context": format_documents(state["documents"]),
            }
        )
        review = review_model.invoke(prompt_value)
        if not isinstance(review, EvidenceReview):
            raise TypeError("證據審查 Agent 未回傳 EvidenceReview")

        # 模型審查負責理解「主張是否受到證據支持」；validate_citations 則用
        # Python 規則確認 draft 中的 chunk_id、source、quote 能否對回檢索文件。
        # 兩種檢查互補：前者做語意判斷，後者做可重現的欄位與原文比對。
        deterministic_issues = validate_citations(state["draft"], state["documents"])
        if deterministic_issues:
            # 只要引用規則檢查發現問題，就強制把 grounded 改成 False，
            # 並將確定性檢查的訊息附加到模型原本列出的 issues。
            review = review.model_copy(
                update={
                    "grounded": False,
                    "issues": review.issues + deterministic_issues,
                }
            )
        return {
            "review": review,
            "trace": state["trace"] + ["證據審查 Agent：檢查主張與引用"],
        }

    def finalize_node(state: MultiAgentState) -> dict:
        # 審查 Agent 有提供 revised_answer 時採用修正版，否則沿用轉譯後文字。
        review = state["review"]
        final_text = review.revised_answer if review.revised_answer else state["candidate"]
        draft = state["draft"]

        # 轉譯和審查階段只處理答案文字，引用仍沿用回答 Agent 產生的 draft.citations。
        # 若草稿原本證據不足，或審查認定內容未受到證據支持，最後都標示證據不足。
        final_answer = RagAnswer(
            answer=final_text,
            citations=draft.citations,
            insufficient_evidence=draft.insufficient_evidence or not review.grounded,
        )
        return {
            "final_answer": final_answer,
            "trace": state["trace"] + ["整理結果：輸出最後答案"],
        }

    # StateGraph 負責定義「有哪些節點」以及「節點按照什麼順序執行」。
    builder = StateGraph(MultiAgentState)

    # add_node 的第一個參數是節點名稱，第二個參數是實際執行的 Python 函式。
    builder.add_node("retrieval_node", retrieval_node)
    builder.add_node("drafting_agent", drafting_agent)
    builder.add_node("teaching_agent", teaching_agent)
    builder.add_node("evidence_agent", evidence_agent)
    builder.add_node("finalize", finalize_node)

    # 這個教學版本故意維持單一路徑。add_edge 定義執行順序：
    # START → 檢索 → 草稿 → 轉譯 → 審查 → 整理 → END。
    builder.add_edge(START, "retrieval_node")
    builder.add_edge("retrieval_node", "drafting_agent")
    builder.add_edge("drafting_agent", "teaching_agent")
    builder.add_edge("teaching_agent", "evidence_agent")
    builder.add_edge("evidence_agent", "finalize")
    builder.add_edge("finalize", END)

    # compile() 把節點與邊組成可用 invoke() 執行的 LangGraph 應用程式。
    return builder.compile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比較檢索、轉譯、審查三個角色如何接力完成回答")
    parser.add_argument("--provider", choices=["gemini", "azure"])
    parser.add_argument("--question", default="校外教學同意書逾期沒交，可以直接算同意嗎？")
    parser.add_argument("--audience", default="國中學生家長")
    parser.add_argument("--show-graph", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Chat Model 負責回答、轉譯與審查；Embedding Model 負責把查詢轉成向量。
    # open_local_vector_store 會開啟前面範例已建立的 .qdrant/ 知識庫。
    model = create_chat_model(args.provider)
    embeddings = create_embeddings(args.provider)
    vector_store, client = open_local_vector_store(embeddings)
    graph = build_multi_agent_graph(model, vector_store)

    try:
        if args.show_graph:
            # draw_mermaid() 輸出 Mermaid 文字，可貼到支援 Mermaid 的工具查看流程圖。
            print(graph.get_graph().draw_mermaid())

        # graph.invoke 的第一個參數是初始 State。documents、draft 等欄位此時尚未產生，
        # 後續節點會依序補入；完成後 result 就是包含所有欄位的最終 State。
        result = graph.invoke(
            {
                "question": args.question,
                "audience": args.audience,
                "trace": [],
            },
            # recursion_limit 限制 LangGraph 最多執行的節點步數，避免流程設定錯誤時無限循環。
            config={"recursion_limit": 12},
        )

        # trace 用來觀察節點順序；review 顯示審查判斷；final_answer 是最後輸出。
        print("執行軌跡：")
        for step in result["trace"]:
            print(f"- {step}")
        print("\n審查結果：")
        print(result["review"].model_dump_json(indent=2))
        print("\n最終回答：")
        print(result["final_answer"].model_dump_json(indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()