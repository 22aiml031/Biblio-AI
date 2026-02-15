"""
LLM-Based Summarization & Insight Generation Module for BiblioAI

- Use Gemma-3 (via HuggingFace or local) for summarization and reasoning
- Input: topic clusters, trend statistics, citation analysis results
- Output: concise, academic, human-readable summaries and insights

Dependencies: transformers (Gemma-3), torch (if required)
"""
import requests
from typing import List, Dict, Any

class Gemma3Summarizer:
    def __init__(self, model_name: str = "gemma:2b", temperature: float = 0.2, api_url: str = "http://localhost:11434/api/generate"):
        """Initialize Gemma-3 summarizer using Ollama local API."""
        self.model_name = model_name
        self.temperature = temperature
        self.api_url = api_url

    def _query_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "temperature": self.temperature
            }
        }
        response = requests.post(self.api_url, json=payload, timeout=120, stream=True)
        response.raise_for_status()
        # Ollama streams responses, so we need to collect the output
        output = ""
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                if '"response":' in data:
                    import json
                    resp_json = json.loads(data)
                    output += resp_json.get("response", "")
        return output.strip()

    def summarize_topic(self, topic_papers: List[Dict[str, Any]], topic_label: str = "") -> str:
        """Generate a summary for a topic cluster."""
        context = "\n".join([paper.get("title", "") for paper in topic_papers])
        prompt = f"Summarize the main research directions and findings for the topic: {topic_label}. Papers: {context}"
        return self._query_ollama(prompt)

    def summarize_trends(self, trend_stats: Dict[str, Any]) -> str:
        """Generate a summary for research trends and gaps."""
        prompt = f"Based on the following statistics, summarize the emerging and declining research trends, and identify any research gaps: {trend_stats}"
        return self._query_ollama(prompt)

# Example usage
if __name__ == "__main__":
    # Example topic cluster
    topic_papers = [
        {"title": "A machine learning-based framework for predicting supply delay risk using big data technology"},
        {"title": "Improving supply chain resilience in the context of pandemic: A perspective of FMCG industries"}
    ]
    summarizer = Gemma3Summarizer()
    print(summarizer.summarize_topic(topic_papers, topic_label="Supply Chain Resilience"))
    # Example trend stats
    trend_stats = {"emerging": ["AI in supply chain"], "declining": ["manual inventory control"], "gaps": ["AI for small-scale suppliers"]}
    print(summarizer.summarize_trends(trend_stats))
