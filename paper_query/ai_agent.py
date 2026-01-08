from __future__ import annotations

import html
import json
import os
import re
import time
from typing import Dict, List, TypedDict
import xml.etree.ElementTree as ET

import requests
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


class Article(TypedDict, total=False):
    title: str
    abstract: str
    translated_title: str
    translated_abstract: str


class SearchResult(TypedDict):
    search_query: str
    count: int
    top2: List[Article]


class AgentState(TypedDict, total=False):
    user_query: str
    is_literature: bool
    answer: str
    search_query: str
    count: int
    top2: List[Article]
    report_path: str
    summary: str


class LogEntry(TypedDict):
    timestamp: str
    user_query: str
    search_query: str
    count: int
    report_path: str


def build_deepseek_chat() -> ChatOpenAI:
    import os
    from langchain_openai import ChatOpenAI

    deepseek_chat = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        temperature=0,
    )
    return deepseek_chat


class PubMedSearchTool:
    def __init__(self) -> None:
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def generate_search_terms(self, user_query: str) -> str:
        return user_query.strip()

    def refine_query(self, query: str, attempt: int) -> str:
        suffix = " review"
        if attempt == 1:
            suffix = " review[pt] AND humans[MeSH Terms]"
        elif attempt == 2:
            suffix = " AND (clinical trial[pt] OR review[pt])"
        return f"{query}{suffix}"

    def _esearch(self, query: str) -> Dict[str, object]:
        response = requests.get(
            f"{self.base_url}esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": 2,
                "retmode": "json",
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json().get("esearchresult", {})
        return {
            "count": int(data.get("count", 0)),
            "ids": data.get("idlist", []),
        }

    def _fetch_titles(self, ids: List[str]) -> Dict[str, str]:
        if not ids:
            return {}
        response = requests.get(
            f"{self.base_url}esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json().get("result", {})
        titles: Dict[str, str] = {}
        for pmid in ids:
            item = result.get(pmid, {})
            titles[pmid] = item.get("title") or "No Title"
        return titles

    def _fetch_abstracts(self, ids: List[str]) -> Dict[str, str]:
        if not ids:
            return {}
        response = requests.get(
            f"{self.base_url}efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
            timeout=30,
        )
        response.raise_for_status()
        abstracts: Dict[str, str] = {}
        root = ET.fromstring(response.text)
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID")
            if not pmid:
                continue
            texts = [
                node.text.strip()
                for node in article.findall(".//AbstractText")
                if node.text and node.text.strip()
            ]
            abstracts[pmid] = " ".join(texts) if texts else "No Abstract"
        return abstracts

    def search(self, user_query: str, retries: int = 3) -> SearchResult:
        query = self.generate_search_terms(user_query)
        for attempt in range(retries):
            result = self._esearch(query)
            count = result["count"]
            ids = result["ids"]
            if count < 5 and attempt < retries - 1:
                query = self.refine_query(query, attempt)
                continue
            titles = self._fetch_titles(ids)
            abstracts = self._fetch_abstracts(ids)
            top2: List[Article] = []
            for pmid in ids:
                top2.append(
                    {
                        "title": titles.get(pmid, "No Title"),
                        "abstract": abstracts.get(pmid, "No Abstract"),
                    }
                )
            return {"search_query": query, "count": count, "top2": top2}
        return {"search_query": query, "count": 0, "top2": []}


class TranslatorTool:
    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm

    def translate(self, text: str) -> str:
        if not text:
            return ""
        response = self.llm.invoke(f"Translate the following text to Chinese:\n\n{text}")
        return response.content


class HTMLReportTool:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _safe_name(self, query: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", query).strip("_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return f"{slug or 'report'}_{timestamp}.html"

    def build_report(self, search_query: str, articles: List[Article]) -> str:
        filename = self._safe_name(search_query)
        output_path = os.path.join(self.output_dir, filename)
        rows = []
        for article in articles:
            rows.append(
                f"""
                <div class="article">
                  <div class="label">English title</div>
                  <div class="text">{html.escape(article.get("title", ""))}</div>
                  <div class="label">English abstract</div>
                  <div class="text">{html.escape(article.get("abstract", ""))}</div>
                  <div class="label">Chinese title</div>
                  <div class="text">{html.escape(article.get("translated_title", ""))}</div>
                  <div class="label">Chinese abstract</div>
                  <div class="text">{html.escape(article.get("translated_abstract", ""))}</div>
                </div>
                """
            )
        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PubMed Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1 {{ font-size: 20px; margin-bottom: 8px; }}
    .query {{ margin-bottom: 16px; color: #444; }}
    .article {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 16px; }}
    .label {{ font-weight: bold; margin-top: 8px; }}
    .text {{ margin: 4px 0 8px 0; }}
  </style>
</head>
<body>
  <h1>Literature Search Report</h1>
  <div class="query">Search query: {html.escape(search_query)}</div>
  {''.join(rows) if rows else '<div>No articles found.</div>'}
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(html_doc)
        return output_path


class AIAgent:
    def __init__(self, report_dir: str = "report", log_path: str = "bak.jsonl") -> None:
        self.search_tool = PubMedSearchTool()
        self.llm = build_deepseek_chat()
        self.translate_tool = TranslatorTool(self.llm)
        self.report_tool = HTMLReportTool(report_dir)
        self.log_path = log_path

    def classify_query(self, state: AgentState) -> AgentState:
        text = (state.get("user_query") or "").lower()
        keywords = (
            "paper",
            "literature",
            "pubmed",
            "article",
            "study",
            "studies",
            "review",
            "trial",
            "treatment",
            "therapy",
            "drug",
            "disease",
            "diagnosis",
        )
        is_literature = any(keyword in text for keyword in keywords)
        update: AgentState = {"is_literature": is_literature}
        if not is_literature:
            update["answer"] = "This does not look like a literature search question."
            update["summary"] = update["answer"]
        return update

    def search_pubmed(self, state: AgentState) -> AgentState:
        user_query = state.get("user_query", "")
        result = self.search_tool.search(user_query)
        return {
            "search_query": result["search_query"],
            "count": result["count"],
            "top2": result["top2"],
        }

    def translate_top2(self, state: AgentState) -> AgentState:
        articles = state.get("top2", [])
        translated: List[Article] = []
        for article in articles:
            translated_title = self.translate_tool.translate(article.get("title", ""))
            translated_abstract = self.translate_tool.translate(article.get("abstract", ""))
            translated.append(
                {
                    **article,
                    "translated_title": translated_title,
                    "translated_abstract": translated_abstract,
                }
            )
        return {"top2": translated}

    def build_report(self, state: AgentState) -> AgentState:
        report_path = self.report_tool.build_report(
            state.get("search_query", ""), state.get("top2", [])
        )
        return {"report_path": report_path}

    def save_log(self, state: AgentState) -> AgentState:
        entry: LogEntry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "user_query": state.get("user_query", ""),
            "search_query": state.get("search_query", ""),
            "count": state.get("count", 0),
            "report_path": state.get("report_path", ""),
        }
        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        summary = (
            f"Search query: {entry['search_query']} | Count: {entry['count']} | "
            f"Report: {entry['report_path']}"
        )
        return {"summary": summary}

    def _route(self, state: AgentState) -> str:
        return "search_pubmed" if state.get("is_literature") else "end"

    def get_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("classify_query", self.classify_query)
        graph.add_node("search_pubmed", self.search_pubmed)
        graph.add_node("translate_top2", self.translate_top2)
        graph.add_node("build_report", self.build_report)
        graph.add_node("save_log", self.save_log)

        graph.set_entry_point("classify_query")
        graph.add_conditional_edges(
            "classify_query",
            self._route,
            {"search_pubmed": "search_pubmed", "end": END},
        )
        graph.add_edge("search_pubmed", "translate_top2")
        graph.add_edge("translate_top2", "build_report")
        graph.add_edge("build_report", "save_log")
        graph.add_edge("save_log", END)
        app = graph.compile()
        return app.get_graph()

    def run(self, user_query: str) -> AgentState:
        app = self._build_app()
        return app.invoke({"user_query": user_query})

    def _build_app(self):
        graph = StateGraph(AgentState)
        graph.add_node("classify_query", self.classify_query)
        graph.add_node("search_pubmed", self.search_pubmed)
        graph.add_node("translate_top2", self.translate_top2)
        graph.add_node("build_report", self.build_report)
        graph.add_node("save_log", self.save_log)

        graph.set_entry_point("classify_query")
        graph.add_conditional_edges(
            "classify_query",
            self._route,
            {"search_pubmed": "search_pubmed", "end": END},
        )
        graph.add_edge("search_pubmed", "translate_top2")
        graph.add_edge("translate_top2", "build_report")
        graph.add_edge("build_report", "save_log")
        graph.add_edge("save_log", END)
        return graph.compile()

