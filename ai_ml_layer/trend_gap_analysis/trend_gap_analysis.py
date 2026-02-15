"""
Trend & Gap Analysis Module for BiblioAI

- Analyze topic-wise publication count over years
- Identify emerging, declining, and saturated topics
- Detect under-explored research gaps

Dependencies: pandas, numpy
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

class TrendGapAnalyzer:
    def __init__(self, clusters: Dict[int, Dict[str, Any]]):
        self.clusters = clusters

    def topic_trend_stats(self) -> pd.DataFrame:
        """Return DataFrame with topic-wise publication count per year and citation stats."""
        records = []
        for cluster_id, meta in self.clusters.items():
            for paper in meta['papers']:
                records.append({
                    'cluster': cluster_id,
                    'year': paper.get('year', None),
                    'citations': paper.get('citations', 0)
                })
        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame()
        trend = df.groupby(['cluster', 'year']).agg(pub_count=('year', 'size'), total_citations=('citations', 'sum')).reset_index()
        return trend

    def identify_trends_gaps(self, min_growth: float = 0.2, stable_margin: float = 0.1, min_gap_size: int = 3, min_citation_velocity: float = 1.0, recent_years: int = 3, min_years: int = 3, min_papers: int = 20) -> Dict[str, Any]:
        """Identify emerging, stable, declining, and gap topics based on publication trends, growth, and citation velocity.
        Adds robust minimum data validation and normalizes growth to avoid false positives on small/weak datasets."""
        trend_df = self.topic_trend_stats()
        if trend_df.empty:
            return {"message": "Insufficient data to infer trends or gaps."}
        summary = {'clusters': [], 'insufficient_data': [], 'insufficient_gaps': False}
        current_year = pd.Timestamp.now().year
        for cluster_id, meta in self.clusters.items():
            cluster_trend = trend_df[trend_df['cluster'] == cluster_id].sort_values('year')
            years = cluster_trend['year'].dropna().astype(int).values
            counts = cluster_trend['pub_count'].values
            citations = cluster_trend['total_citations'].sum()
            n_papers = sum(counts)
            unique_years = np.unique(years)
            years_active = years[-1] - years[0] + 1 if len(years) > 1 else 1
            citation_velocity = citations / (n_papers * years_active) if n_papers > 0 and years_active > 0 else 0
            # Minimum data validation
            if len(unique_years) < min_years or n_papers < min_papers:
                trend = 'STABLE'
                insufficient = True
            else:
                insufficient = False
                # Smoothed/log growth calculation
                recent_mask = years >= (current_year - recent_years)
                prev_mask = years < (current_year - recent_years)
                recent_counts = counts[recent_mask] if any(recent_mask) else np.array([])
                prev_counts = counts[prev_mask] if any(prev_mask) else np.array([])
                recent_sum = np.sum(recent_counts)
                prev_sum = np.sum(prev_counts)
                # Use log(1 + count) smoothing
                recent_log = np.log1p(recent_sum)
                prev_log = np.log1p(prev_sum)
                if prev_log == 0 and recent_log > 0:
                    growth_rate = 1.0
                elif prev_log == 0:
                    growth_rate = 0.0
                else:
                    growth_rate = (recent_log - prev_log) / prev_log
                # Trend classification
                if growth_rate > min_growth and citation_velocity > 0:
                    trend = 'EMERGING'
                elif growth_rate < -stable_margin and citation_velocity > 0:
                    trend = 'DECLINING'
                else:
                    trend = 'STABLE'
            # Topic name from top keywords
            top_keywords = meta.get('top_keywords', [])
            topic_name = ', '.join(top_keywords[:3]) if top_keywords else f"Cluster {cluster_id}"
            # Gap detection: only if sufficient temporal data
            if len(unique_years) >= min_years and n_papers >= min_gap_size:
                is_gap = (n_papers < min_papers and citation_velocity > min_citation_velocity and years[-1] >= current_year - recent_years)
            else:
                is_gap = False
            cluster_info = {
                'cluster_id': int(cluster_id),
                'topic_name': topic_name,
                'trend': trend,
                'growth_rate': round(growth_rate, 3) if not insufficient else None,
                'citation_velocity': round(citation_velocity, 3),
                'years_active': int(years_active),
                'pub_counts_per_year': {int(y): int(c) for y, c in zip(years, counts)},
                'gap': is_gap,
                'n_papers': int(n_papers),
                'total_citations': int(citations),
                'insufficient_data': insufficient
            }
            summary['clusters'].append(cluster_info)
            if insufficient:
                summary['insufficient_data'].append(cluster_id)
        # For convenience, also output lists of cluster_ids for each trend/gap
        summary['emerging'] = [c['cluster_id'] for c in summary['clusters'] if c['trend'] == 'EMERGING']
        summary['declining'] = [c['cluster_id'] for c in summary['clusters'] if c['trend'] == 'DECLINING']
        summary['stable'] = [c['cluster_id'] for c in summary['clusters'] if c['trend'] == 'STABLE']
        summary['gaps'] = [c['cluster_id'] for c in summary['clusters'] if c['gap']]
        # If all clusters have insufficient data, mark gaps as not inferable
        if len(summary['insufficient_data']) == len(self.clusters):
            summary['insufficient_gaps'] = True
        return summary

# Example usage
if __name__ == "__main__":
    # Example clusters dict
    clusters = {
        0: {'papers': [{'year': 2020, 'citations': 5}, {'year': 2021, 'citations': 10}]},
        1: {'papers': [{'year': 2020, 'citations': 2}, {'year': 2021, 'citations': 1}]},
        2: {'papers': [{'year': 2021, 'citations': 0}]}
    }
    analyzer = TrendGapAnalyzer(clusters)
    print(analyzer.topic_trend_stats())
    print(analyzer.identify_trends_gaps())
