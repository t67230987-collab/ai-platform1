"""
多智能体协作系统 - Multi-Agent Collaboration
多个 AI 角色同时分析同一问题, 综合得出最优解
"""
from src.llm_client import LLMClient
from config import DEFAULT_TEMPERATURE


# 预定义 Agent 角色
PRESET_AGENTS = [
    {
        "name": "🔬 技术专家",
        "role": "你是一位资深技术专家，擅长从技术实现角度分析问题。请给出具体、可操作的技术方案。",
    },
    {
        "name": "💼 产品经理",
        "role": "你是一位经验丰富的产品经理，关注用户需求、产品价值和市场可行性。",
    },
    {
        "name": "🎨 创意设计师",
        "role": "你是一位富有创意的设计师，善于提出新颖的想法和用户体验优化方案。",
    },
    {
        "name": "🔍 批判性思维者",
        "role": "你是一位严谨的批判性思维者，负责找出方案中的漏洞、风险和不足之处。",
    },
]


class MultiAgentSystem:
    """多智能体协作系统"""

    def __init__(self, backend="ollama", model=None):
        self.backend = backend
        self.model = model
        self.agents = PRESET_AGENTS

    def run(self, question, enabled_agents=None, temperature=DEFAULT_TEMPERATURE,
            max_tokens=512, progress_callback=None):
        """
        多 Agent 并行分析问题
        - enabled_agents: 启用的 Agent 索引列表, None=全部启用
        - progress_callback: 进度回调 (agent_name, response)
        """
        if enabled_agents is None:
            enabled_agents = list(range(len(self.agents)))

        responses = []
        for i in enabled_agents:
            if i >= len(self.agents):
                continue
            agent = self.agents[i]
            client = LLMClient(self.backend, self.model)

            prompt = f"{agent['role']}\n\n问题: {question}\n\n请从你的专业角度给出分析和建议。"

            response = client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            result = {"agent": agent["name"], "response": response}
            responses.append(result)

            if progress_callback:
                progress_callback(agent["name"], response)

        return responses

    def synthesize(self, question, agent_responses, temperature=DEFAULT_TEMPERATURE,
                   max_tokens=1024):
        """
        综合各 Agent 观点，生成最终答案
        """
        if not agent_responses:
            return "没有 Agent 响应，无法综合。"

        # 拼接各方观点
        opinions = "\n\n".join([
            f"【{r['agent']}】:\n{r['response']}"
            for r in agent_responses
        ])

        client = LLMClient(self.backend, self.model)
        synth_prompt = (
            f"你是一位综合决策者。以下是多位专家对同一个问题的分析:\n\n"
            f"## 原始问题\n{question}\n\n"
            f"## 专家意见\n{opinions}\n\n"
            f"请你综合各方观点，给出一个全面、平衡的最终答案。"
        )

        return client.chat(
            messages=[{"role": "user", "content": synth_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
