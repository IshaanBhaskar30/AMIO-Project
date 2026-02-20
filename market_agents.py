# ============================================================
# market_agents.py
# Autonomous Market Intelligence Orchestrator (AMIO)
# Production-Grade Version
# ============================================================

from typing import Annotated, TypedDict, List, Optional
import operator
from langgraph.graph import START, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from tenacity import retry, stop_after_attempt, wait_fixed
import os
import json


# ============================================================
# GRAPH STATE
# ============================================================

class MarketState(TypedDict):
    query: str
    plan: Optional[dict]
    overview: Optional[str]
    trends: Optional[str]
    financials: Optional[str]
    competitors: Optional[str]
    risks: Optional[str]
    sources: Annotated[List[str], operator.add]  # auto-merge
    final_report: Optional[dict]
    confidence: Optional[dict]
    refinement_count: int


# ============================================================
# RETRY WRAPPER
# ============================================================

@retry(stop=stop_after_attempt(2), wait=wait_fixed(2))
def safe_invoke(llm, messages):
    return llm.invoke(messages)


# ============================================================
# GRAPH BUILDER
# ============================================================

def build_graph(groq_key: str, tavily_key: str, model_name: str):

    os.environ["TAVILY_API_KEY"] = tavily_key

    primary_llm = ChatGroq(
        model=model_name,
        api_key=groq_key,
        temperature=0.2
    )

    search_tool = TavilySearchResults(
        tavily_api_key=tavily_key,
        max_results=3
    )

    # ========================================================
    # PLANNER
    # ========================================================

    def planner(state: MarketState):

        prompt = f"""
        Decompose this intelligence query into structured needs.

        Query:
        {state["query"]}

        Return ONLY valid JSON:
        {{
          "entity": "...",
          "need_overview": true,
          "need_trends": true,
          "need_financials": true,
          "need_competitors": true,
          "need_risks": true
        }}
        """

        response = safe_invoke(primary_llm, [HumanMessage(content=prompt)])

        try:
            plan_dict = json.loads(response.content)
        except:
            plan_dict = {
                "entity": state["query"],
                "need_overview": True,
                "need_trends": True,
                "need_financials": True,
                "need_competitors": True,
                "need_risks": True
            }

        return {"plan": plan_dict}

    # ========================================================
    # ROUTER
    # ========================================================

    def route_tasks(state: MarketState):
        plan = state["plan"]
        tasks = []

        if plan.get("need_overview"):
            tasks.append("overview_node")
        if plan.get("need_trends"):
            tasks.append("trends_node")
        if plan.get("need_financials"):
            tasks.append("financials_node")
        if plan.get("need_competitors"):
            tasks.append("competitors_node")
        if plan.get("need_risks"):
            tasks.append("risks_node")

        return tasks

    # ========================================================
    # SEARCH AGENT FACTORY
    # ========================================================

    def make_search_agent(field_name, query_suffix):

        def agent(state: MarketState):

            results = search_tool.invoke(
                f"{state['plan']['entity']} {query_suffix}"
            )

            text = "\n".join([r["content"] for r in results])
            urls = [r["url"] for r in results if "url" in r]

            return {
                field_name: text,
                "sources": urls  # DO NOT manually merge
            }

        return agent

    overview_agent = make_search_agent("overview", "company overview")
    trends_agent = make_search_agent("trends", "industry trends 2024")
    financial_agent = make_search_agent("financials", "financial performance revenue profit")
    competitor_agent = make_search_agent("competitors", "main competitors market share")
    risk_agent = make_search_agent("risks", "regulatory geopolitical operational risks")

    # ========================================================
    # AGGREGATOR
    # ========================================================

    def aggregator(state: MarketState):

        synthesis_prompt = f"""
        Create an institutional-grade executive report.

        Overview:
        {state.get("overview")}

        Trends:
        {state.get("trends")}

        Financials:
        {state.get("financials")}

        Competitors:
        {state.get("competitors")}

        Risks:
        {state.get("risks")}

        Format clearly with headings and bullet points.
        """

        response = safe_invoke(primary_llm, [HumanMessage(content=synthesis_prompt)])

        report = {
            "content": response.content,
            "sources": list(set(state.get("sources", [])))
        }

        return {"final_report": report}

    # ========================================================
    # EVALUATOR
    # ========================================================

    def evaluator(state: MarketState):

        evaluation_prompt = f"""
        Evaluate this executive report from 0 to 100.

        Report:
        {state["final_report"]["content"]}

        Sources Count: {len(state.get("sources", []))}

        Return ONLY valid JSON:
        {{
          "score": 0-100,
          "reasoning": "...",
          "missing_elements": ["..."]
        }}
        """

        response = safe_invoke(primary_llm, [HumanMessage(content=evaluation_prompt)])

        try:
            confidence = json.loads(response.content)
        except:
            confidence = {
                "score": 85,
                "reasoning": "Fallback confidence.",
                "missing_elements": []
            }

        return {"confidence": confidence}

    # ========================================================
    # REFINEMENT
    # ========================================================

    def refinement_router(state: MarketState):
        if state["confidence"]["score"] < 70 and state["refinement_count"] < 1:
            return "refine"
        return END

    def refine_node(state: MarketState):

        improve_prompt = f"""
        Improve this executive report by addressing:

        {state["confidence"]["missing_elements"]}

        Original Report:
        {state["final_report"]["content"]}
        """

        response = safe_invoke(primary_llm, [HumanMessage(content=improve_prompt)])

        improved_report = {
            "content": response.content,
            "sources": list(set(state.get("sources", [])))
        }

        return {
            "final_report": improved_report,
            "refinement_count": state["refinement_count"] + 1
        }

    # ========================================================
    # GRAPH CONSTRUCTION
    # ========================================================

    graph = StateGraph(MarketState)

    graph.add_node("planner", planner)
    graph.add_node("overview_node", overview_agent)
    graph.add_node("trends_node", trends_agent)
    graph.add_node("financials_node", financial_agent)
    graph.add_node("competitors_node", competitor_agent)
    graph.add_node("risks_node", risk_agent)
    graph.add_node("aggregator", aggregator)
    graph.add_node("evaluator", evaluator)
    graph.add_node("refine", refine_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_tasks)

    graph.add_edge("overview_node", "aggregator")
    graph.add_edge("trends_node", "aggregator")
    graph.add_edge("financials_node", "aggregator")
    graph.add_edge("competitors_node", "aggregator")
    graph.add_edge("risks_node", "aggregator")

    graph.add_edge("aggregator", "evaluator")
    graph.add_conditional_edges("evaluator", refinement_router)

    graph.add_edge("refine", "aggregator")

    return graph.compile()
