"""校園 Agent 範例使用的虛構資料與查詢函式。

資料直接放在 Python 變數中，不使用資料庫，也不包含真實師生資訊。
"""

from __future__ import annotations

TEACHER_SCHEDULES: dict[str, tuple[str, ...]] = {
    "王怡婷": (
        "星期一：第 1 節五年一班國語、第 3 節五年二班國語",
        "星期二：第 2 節五年一班國語、第 5 節閱讀社團",
        "星期三：第 1 節五年二班國語、第 4 節五年一班閱讀",
        "星期四：第 2 節五年二班國語、第 6 節班級共讀",
        "星期五：第 3 節五年一班國語、第 5 節五年二班閱讀",
    ),
    "王志明": (
        "星期一：第 2 節五年一班數學、第 5 節五年二班數學",
        "星期二：第 1 節五年二班數學、第 4 節五年一班數學",
        "星期三：第 2 節五年一班數學",
        "星期四：第 1 節五年二班數學、第 5 節五年一班數學",
        "星期五：第 2 節五年二班數學、第 4 節數學補救教學",
    ),
    "林美玲": (
        "星期一：第 3 節五年一班英語、第 6 節五年二班英語",
        "星期二：第 3 節五年二班英語",
        "星期三：第 3 節五年一班英語、第 4 節五年二班英語",
        "星期四：第 3 節五年一班英語",
        "星期五：第 1 節五年二班英語、第 6 節英語社團",
    ),
}

ACTIVITY_PROCEDURES: dict[str, tuple[str, ...]] = {
    "校外教學": (
        "提出活動計畫與學習目標",
        "完成家長同意書與參加名冊",
        "辦理交通、保險與緊急聯絡規劃",
        "依校內程序核准後再通知集合資訊",
    ),
    "校內閱讀活動": (
        "確認活動對象、人數與帶隊教師",
        "向場地管理單位提出借用申請",
        "準備閱讀材料與器材清單",
        "公告時間、地點與注意事項",
    ),
    "親師座談": (
        "確認班級、日期與預估出席人數",
        "安排場地與報到方式",
        "彙整討論議題並通知家長",
        "會後保存簽到與會議紀錄",
    ),
}

CAMPUS_VENUES: dict[str, dict[str, object]] = {
    "圖書館共讀區": {
        "capacity": 40,
        "available_periods": ("星期二下午", "星期三下午", "星期五上午"),
    },
    "視聽教室": {
        "capacity": 80,
        "available_periods": ("星期二下午", "星期四下午", "星期五下午"),
    },
    "活動中心": {
        "capacity": 300,
        "available_periods": ("星期三下午", "星期四下午", "星期五下午"),
    },
}


def query_teacher_schedule(teacher_name: str) -> str:
    """以老師姓名關鍵字查詢一週課表，回傳所有符合的老師。"""
    keyword = teacher_name.strip()
    if not keyword:
        return "請輸入老師姓名或部分姓名，例如：王、怡婷、林美玲。"

    matches = [
        (name, schedule)
        for name, schedule in TEACHER_SCHEDULES.items()
        if keyword in name
    ]
    if not matches:
        return f"查無姓名包含「{keyword}」的老師課表。"

    blocks = []
    for name, schedule in matches:
        blocks.append(f"{name}老師\n" + "\n".join(f"- {item}" for item in schedule))
    return "\n\n".join(blocks)


def query_activity_procedure(activity_type: str) -> str:
    """以活動類型關鍵字查詢校內申請流程。"""
    keyword = activity_type.strip()
    if not keyword:
        return "請輸入活動類型，例如：校外教學、閱讀活動、親師座談。"

    matches = [
        (name, steps)
        for name, steps in ACTIVITY_PROCEDURES.items()
        if keyword in name or name in keyword
    ]
    if not matches:
        available = "、".join(ACTIVITY_PROCEDURES)
        return f"查無「{keyword}」的活動流程。目前可查詢：{available}。"

    blocks = []
    for name, steps in matches:
        formatted_steps = "\n".join(
            f"{index}. {step}" for index, step in enumerate(steps, 1)
        )
        blocks.append(f"{name}申請流程\n{formatted_steps}")
    return "\n\n".join(blocks)


def query_campus_venue(expected_people: int, period: str = "") -> str:
    """依預估人數與時段查詢容量足夠的虛構校園場地。"""
    if expected_people <= 0:
        return "預估人數必須大於 0。"

    requested_period = period.strip()
    matches = []
    for name, venue in CAMPUS_VENUES.items():
        capacity = int(venue["capacity"])
        available_periods = tuple(str(item) for item in venue["available_periods"])
        if capacity < expected_people:
            continue
        if requested_period and requested_period not in available_periods:
            continue
        matches.append((name, capacity, available_periods))

    if not matches:
        period_text = f"且可於{requested_period}使用" if requested_period else ""
        return f"查無可容納 {expected_people} 人{period_text}的場地。"

    lines = [f"可容納 {expected_people} 人的場地："]
    for name, capacity, available_periods in matches:
        lines.append(f"- {name}（容量 {capacity} 人；可用時段：{'、'.join(available_periods)}）")
    return "\n".join(lines)


def calculate_activity_budget(
    participant_count: int,
    transportation_cost: int = 0,
    meal_cost_per_person: int = 0,
    material_cost_per_person: int = 0,
) -> str:
    """試算活動交通、餐費與材料費，不代表正式核銷金額。"""
    amounts = (
        participant_count,
        transportation_cost,
        meal_cost_per_person,
        material_cost_per_person,
    )
    if participant_count <= 0:
        return "參加人數必須大於 0。"
    if any(amount < 0 for amount in amounts[1:]):
        return "費用不可為負數。"

    meal_total = participant_count * meal_cost_per_person
    material_total = participant_count * material_cost_per_person
    total = transportation_cost + meal_total + material_total
    return (
        "活動費用試算\n"
        f"- 交通費：{transportation_cost:,} 元\n"
        f"- 餐費：{participant_count} 人 x {meal_cost_per_person:,} 元 = {meal_total:,} 元\n"
        f"- 材料費：{participant_count} 人 x {material_cost_per_person:,} 元 "
        f"= {material_total:,} 元\n"
        f"- 合計：{total:,} 元\n"
        "本結果僅供課程試算，實際金額仍須依學校採購與核銷程序確認。"
    )
