"""
Test data: 多跳推理、时间推理、跨会话记忆
每个场景包含：对话记录 + 问题 + 标准答案
"""
import json

TEST_CASES = {
    "multi_hop": [
        {
            "id": "mh1",
            "name": "两跳：人物关系 + 事件",
            "conversation": [
                "Alice: 我昨天去了Nate的生日派对，人挺多的。",
                "Bob: Nate最近在忙什么？",
                "Alice: 他在准备周末的电竞比赛，他是战队的主力。",
                "Bob: 那Caroline去吗？",
                "Alice: Caroline去度假了，七月份才回来。",
                "Bob: Jonna呢？我记得她也玩游戏的。",
                "Alice: Jonna最近在写剧本，投了一个电影工作室。",
                "Bob: 结果怎么样？",
                "Alice: 被拒了，她说下次再试试。",
            ],
            "question": "Nate这个周末要参加什么活动？",
            "answer": "电竞比赛",
            "category": "entity_relation",
        },
        {
            "id": "mh2",
            "name": "两跳：间接关系",
            "conversation": [
                "User: 我想学做咖啡拉花。",
                "Assistant: 推荐你去看看YouTube上的Latte Art Channel，有很多教程。",
                "User: 好的，我昨晚看了，但牛奶打发总是失败。",
                "Assistant: 牛奶要用全脂的，温度控制在60-65度。",
                "User: 我用的就是全脂牛奶，难道是温度计不准？",
                "Assistant: 可以换一个数字温度计试试。",
                "User: 对了，我还买了他们的咖啡豆推荐，Ethiopia Yirgacheffe。",
            ],
            "question": "用户做咖啡拉花遇到的具体问题是什么？",
            "answer": "牛奶打发总是失败",
            "category": "issue_diagnosis",
        },
        {
            "id": "mh3",
            "name": "三跳：多步关联",
            "conversation": [
                "小王: 听说张老师要退休了？",
                "小李: 对，年底就退，他在物理系教了30年。",
                "小王: 那谁接他的量子力学课？",
                "小李: 好像是陈教授，他刚从MIT回来。",
                "小王: 陈教授研究方向是什么？",
                "小李: 量子计算，特别是拓扑量子纠错。\n他去年发了一篇Nature Physics。",
                "小王: 那学生有福了，现在量子计算这么火。",
                "小李: 是啊，他还带了个团队做超导量子比特。",
            ],
            "question": "接替张老师量子力学课的陈教授，他的研究方向和最近的代表性成果是什么？",
            "answer": "量子计算，拓扑量子纠错，去年发了Nature Physics",
            "category": "multi_step",
        },
        {
            "id": "mh4",
            "name": "否定排除推理",
            "conversation": [
                "甲: 下周的团建大家想去哪？",
                "乙: 我想去漂流，夏天玩水爽。",
                "丙: 漂流太危险了吧，上次有人受伤。",
                "丁: 那去郊野公园烧烤？",
                "甲: 天气预报说下周有雨，烧烤不太行。",
                "丙: 密室逃脱怎么样？我们公司附近新开了一家。",
                "乙: 可以可以，我记得叫Escape Lab。",
                "丁: 好，那就定密室逃脱吧。",
            ],
            "question": "团建最后没有选择漂流的原因是什么？",
            "answer": "因为漂流被认为太危险，上次有人受伤。",
            "category": "negation_reasoning",
        },
    ],
    "temporal": [
        {
            "id": "tm1",
            "name": "事件排序",
            "conversation": [
                "2024-03-01 用户: 我今天开始健身了。",
                "2024-03-15 用户: 健身两周了，瘦了2斤。",
                "2024-04-01 用户: 健身一个月，瘦了5斤，开心！",
                "2024-04-10 用户: 膝盖有点疼，去看了医生说是跑步姿势不对。",
                "2024-04-15 用户: 休息了几天，改成游泳了。",
                "2024-05-01 用户: 游泳三周了，膝盖不疼了。",
            ],
            "question": "用户停止跑步开始游泳的原因是什么？",
            "answer": "因为跑步姿势不对导致膝盖疼，医生建议后转为游泳",
            "category": "temporal_causal",
        },
        {
            "id": "tm2",
            "name": "时间区间",
            "conversation": [
                "2024-01-10 Tom: 我刚买了一台MacBook Pro M3。",
                "2024-02-05 Tom: 用了三周了，电池续航真不错。",
                "2024-03-20 Tom: 发现一个bug，屏幕偶尔闪一下。",
                "2024-04-01 Tom: 预约了天才吧，下周去修。",
                "2024-04-08 Tom: 去天才吧换了台新机，店员很爽快。",
            ],
            "question": "Tom从发现屏幕问题到去天才吧维修，中间隔了多久？",
            "answer": "大约一周（3月20日发现，4月1日预约，4月8日去修）",
            "category": "interval",
        },
        {
            "id": "tm3",
            "name": "跨时间关联",
            "conversation": [
                "2024-06-01 用户: 给爸妈买了海南航空6月20号的机票，让他们来北京玩。",
                "2024-06-05 用户: 妈说想去看故宫，我预约了6月22号的票。",
                "2024-06-15 用户: 查了一下天气预报，6月20-25号北京都有雨。",
                "2024-06-20 用户: 爸妈到了，去接机了。",
                "2024-06-22 用户: 下雨天逛故宫，别有一番风味。",
            ],
            "question": "用户父母是什么时候到的北京？",
            "answer": "2024年6月20日",
            "category": "exact_date",
        },
    ],
    "cross_session": [
        {
            "id": "cs1",
            "name": "跨会话偏好聚合",
            "conversations": [
                [
                    "用户: 最近想学一个新技能，有什么推荐？",
                    "助理: 看你对什么感兴趣了，编程、乐器、语言都可以。",
                    "用户: 我大学学过一点吉他，但早忘了。",
                ],
                [
                    "用户: 看到同事在学Python，我也在想要不要试试。",
                    "助理: Python入门不难，而且应用很广。",
                    "用户: 但我觉得音乐类的可能更适合我。",
                ],
                [
                    "用户: 我决定重新学吉他了，已经买了一把雅马哈。",
                    "助理: 不错的选择！建议从简单的流行歌开始。",
                    "用户: 嗯，我先练一周C大调音阶。",
                ],
            ],
            "question": "用户最终决定学的技能是什么？为什么选择了这个而不是编程？",
            "answer": "吉他（重新学），因为觉得音乐类的更适合自己",
            "category": "preference_aggregation",
        },
        {
            "id": "cs2",
            "name": "跨会话事实合并",
            "conversations": [
                [
                    "用户: 我住的公寓管理费每月1500。",
                    "助理: 那房子是你自己的还是租的？",
                    "用户: 租的，月租8000，在九号公馆。",
                ],
                [
                    "用户: 最近想养只猫。",
                    "助理: 你住的地方允许养宠物吗？",
                    "用户: 管理比较严，不让养。",
                ],
                [
                    "用户: 打算搬家了，想找个能养猫的地方。",
                    "助理: 预算是多少？",
                    "用户: 总预算1万以内吧，含管理费。",
                ],
            ],
            "question": "用户目前月租加管理费一共多少？想搬家找新房子，预算上限是多少？",
            "answer": "目前月租8000+管理费1500=9500；搬家预算总上限1万",
            "category": "fact_merge",
        },
        {
            "id": "cs3",
            "name": "跨会话状态追踪",
            "conversations": [
                [
                    "小明: 我下周要面谷歌，紧张。",
                    "朋友: 准备得怎么样了？",
                    "小明: 刷了200道LeetCode，系统设计还没看。",
                ],
                [
                    "小明: 谷歌面完了，算法过了，系统设计答得一般。",
                    "朋友: 那有后续吗？",
                    "小明: 说这周五给结果。",
                ],
                [
                    "小明: 谷歌给了L4 offer！包裹还没谈。",
                    "朋友: 恭喜！其他公司还有在面的吗？",
                    "小明: 亚马逊下周onsite，不过应该不会去了。",
                ],
            ],
            "question": "小明谷歌面试的最终结果是什么？他为此做了哪些准备？",
            "answer": "拿到L4 offer；刷了200道LeetCode，系统设计准备不足但算法过了",
            "category": "state_tracking",
        },
    ],
}


def format_conversation_for_mragent(conv):
    """Format conversation for MRAgent-style processing"""
    lines = []
    for i, line in enumerate(conv):
        lines.append(f"Turn {i}: {line}")
    return "\n".join(lines)


def format_conversations_for_mragent(convs):
    """Format multi-session conversations"""
    all_lines = []
    for s_idx, session in enumerate(convs):
        all_lines.append(f"=== Session {s_idx} ===")
        for i, line in enumerate(session):
            all_lines.append(f"Turn {i}: {line}")
    return "\n".join(all_lines)


if __name__ == "__main__":
    total = sum(len(v) for v in TEST_CASES.values())
    print(f"Total test cases: {total}")
    for cat, cases in TEST_CASES.items():
        print(f"  {cat}: {len(cases)} cases")
        for c in cases:
            print(f"    [{c['id']}] {c['name']}")
