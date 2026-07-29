"""Single-Agent 與 Multi-Agent 共用的校園查詢工具。"""

from __future__ import annotations

from langchain.tools import tool

from examples.campus_data import (
    calculate_activity_budget,
    query_activity_procedure,
    query_campus_venue,
    query_teacher_schedule,
)


@tool
def search_teacher_schedule(teacher_name: str) -> str:
    """查詢虛構教師的一週課表。

    Args:
        teacher_name: 教師完整姓名或部分姓名，例如「王怡婷」或「王」。
    """
    return query_teacher_schedule(teacher_name)


@tool
def search_activity_procedure(activity_type: str) -> str:
    """查詢校園活動的申請與準備流程。

    Args:
        activity_type: 活動類型，例如「校外教學」、「閱讀活動」或「親師座談」。
    """
    return query_activity_procedure(activity_type)


@tool
def search_campus_venue(expected_people: int, period: str = "") -> str:
    """依人數與時段查詢容量足夠的虛構校園場地。

    Args:
        expected_people: 預估使用場地的人數。
        period: 希望使用的時段，例如「星期五下午」；未指定時可留空。
    """
    return query_campus_venue(expected_people, period)


@tool
def estimate_activity_budget(
    participant_count: int,
    transportation_cost: int = 0,
    meal_cost_per_person: int = 0,
    material_cost_per_person: int = 0,
) -> str:
    """試算校園活動的交通費、餐費、材料費與合計。

    Args:
        participant_count: 參加活動的總人數。
        transportation_cost: 交通費總額，未提供時為 0。
        meal_cost_per_person: 每人餐費，未提供時為 0。
        material_cost_per_person: 每人材料費，未提供時為 0。
    """
    return calculate_activity_budget(
        participant_count=participant_count,
        transportation_cost=transportation_cost,
        meal_cost_per_person=meal_cost_per_person,
        material_cost_per_person=material_cost_per_person,
    )


CAMPUS_TOOLS = [
    search_teacher_schedule,
    search_activity_procedure,
    search_campus_venue,
    estimate_activity_budget,
]