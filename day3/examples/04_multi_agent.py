"""使用 LangGraph 協調教務、學務與總務 Agent。

執行：uv run examples/04_multi_agent.py
"""

from __future__ import annotations

import argparse
from operator import add
from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from examples.campus_tools import (
    estimate_activity_budget,
    search_activity_procedure,
    search_campus_venue,
    search_teacher_schedule,
)
from examples.message_utils import content_to_text, last_message_text
from examples.model_factory import create_chat_model

ExpertName = Literal["teaching_affairs", "student_affairs", "general_affairs"]

EXPERT_LABELS: dict[ExpertName, str] = {
    "teaching_affairs": "教務 Agent",
    "student_affairs": "學務 Agent",
    "general_affairs": "總務 Agent",
}

DEFAULT_QUESTION = "請查王怡婷老師星期二的課表"


class RouteDecision(BaseModel):
    destinations: list[ExpertName] = Field(
        min_length=1,
        description="回答問題所需的一到多位專家，不可加入清單以外的名稱",
    )
    reason: str = Field(description="簡短說明分派理由")


class CampusState(TypedDict):
    question: str
    destinations: list[ExpertName]
    route_reason: str
    expert_results: Annotated[list[str], add]
    final_answer: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph 校園 Multi-Agent")
    parser.add_argument("--provider", choices=["gemini", "azure"])
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    return parser.parse_args()


def build_workflow(model: BaseChatModel):
    """建立 Router、三位專家與整合節點組成的 LangGraph。"""
    router = model.with_structured_output(RouteDecision)
    teaching_agent = create_agent(
        model=model,
        tools=[search_teacher_schedule],
        system_prompt=(
            "你是教務 Agent，只處理教師課表。必須使用課表工具，"
            "並且只能整理工具實際回傳的授課資訊。不要回答場地、活動流程或費用，"
            "也不得推測課表未列出的時段可以自由安排。"
        ),
        name="teaching_affairs_agent",
    )
    student_agent = create_agent(
        model=model,
        tools=[search_activity_procedure, estimate_activity_budget],
        system_prompt=(
            "你是學務 Agent，只處理活動申請流程與費用試算。"
            "必須使用對應工具，並且只能整理工具實際回傳的內容。"
            "不要回答教師課表或建議場地，未提供費用時也不要自行假設金額。"
        ),
        name="student_affairs_agent",
    )
    general_agent = create_agent(
        model=model,
        tools=[search_campus_venue],
        system_prompt=(
            "你是總務 Agent，只處理場地容量與時段。必須使用場地工具，"
            "並且只能整理工具實際回傳的場地。不要回答教師課表、活動流程或費用，"
            "工具未列出的場地不可自行補充。"
        ),
        name="general_affairs_agent",
    )

    def route_node(state: CampusState) -> dict[str, object]:
        decision = router.invoke(
            [
                (
                    "system",
                    "你是校務問題協調者。依問題選擇所需專家："
                    "teaching_affairs 查教師課表；"
                    "student_affairs 查活動流程或試算費用；"
                    "general_affairs 查場地。複合問題必須選擇多位專家。",
                ),
                ("human", state["question"]),
            ]
        )
        if not isinstance(decision, RouteDecision):
            raise TypeError("路由模型未回傳 RouteDecision")
        destinations = list(dict.fromkeys(decision.destinations))
        return {"destinations": destinations, "route_reason": decision.reason}

    def dispatch_experts(state: CampusState) -> list[Send]:
        return [
            Send(destination, {"question": state["question"]})
            for destination in state["destinations"]
        ]

    def teaching_node(state: CampusState) -> dict[str, list[str]]:
        result = teaching_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"只處理下列問題的教師課表部分：{state['question']}",
                    }
                ]
            },
            config={"recursion_limit": 8},
        )
        return {"expert_results": [f"【教務 Agent】\n{last_message_text(result)}"]}

    def student_node(state: CampusState) -> dict[str, list[str]]:
        result = student_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"只處理下列問題的活動流程或費用部分：{state['question']}",
                    }
                ]
            },
            config={"recursion_limit": 8},
        )
        return {"expert_results": [f"【學務 Agent】\n{last_message_text(result)}"]}

    def general_node(state: CampusState) -> dict[str, list[str]]:
        result = general_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"只處理下列問題的場地部分：{state['question']}",
                    }
                ]
            },
            config={"recursion_limit": 8},
        )
        return {"expert_results": [f"【總務 Agent】\n{last_message_text(result)}"]}

    def integrate_node(state: CampusState) -> dict[str, str]:
        source_text = "\n\n".join(state["expert_results"])
        response = model.invoke(
            [
                (
                    "system",
                    "你是校務整合者。只能整理專家結果，不可新增其中沒有的資料。"
                    "若專家指出限制或資料不足，必須保留。最後提醒這是課程虛構資料。",
                ),
                (
                    "human",
                    f"原始問題：{state['question']}\n\n專家結果：\n{source_text}",
                ),
            ]
        )
        return {"final_answer": content_to_text(response.content)}

    builder = StateGraph(CampusState)
    builder.add_node("route", route_node)
    builder.add_node("teaching_affairs", teaching_node)
    builder.add_node("student_affairs", student_node)
    builder.add_node("general_affairs", general_node)
    builder.add_node("integrate", integrate_node)
    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", dispatch_experts)
    builder.add_edge("teaching_affairs", "integrate")
    builder.add_edge("student_affairs", "integrate")
    builder.add_edge("general_affairs", "integrate")
    builder.add_edge("integrate", END)
    return builder.compile()


def main() -> None:
    args = parse_args()
    model = create_chat_model(args.provider)
    workflow = build_workflow(model)
    result = workflow.invoke(
        {
            "question": args.question,
            "destinations": [],
            "route_reason": "",
            "expert_results": [],
            "final_answer": "",
        }
    )

    labels = [EXPERT_LABELS[destination] for destination in result["destinations"]]
    print(f"問題：{args.question}")
    print(f"路由：{'、'.join(labels)}")
    print(f"理由：{result['route_reason']}\n")
    print("=== 專家結果 ===")
    print("\n\n".join(result["expert_results"]))
    print("\n=== 整合回答 ===")
    print(result["final_answer"])


if __name__ == "__main__":
    main()