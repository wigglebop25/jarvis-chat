"""Mood analysis and diagnostics commands."""

import logging

logger = logging.getLogger(__name__)


class MoodCommands:
    """Commands for mood analysis diagnostics."""
    
    @staticmethod
    async def run_mood_analysis() -> str:
        """Run mood correlation analysis immediately."""
        try:
            from ...rag.mood_integration import periodically_update_mood_correlations
            
            await periodically_update_mood_correlations(force=True)
            return "✓ Mood correlation analysis completed"
            
        except Exception as e:
            logger.error(f"Failed to run mood analysis: {e}")
            return f"✗ Error: {e}"
    
    @staticmethod
    def show_mood_correlations() -> str:
        """Show discovered mood correlations."""
        try:
            from ...rag import get_mood_analyzer
            
            analyzer = get_mood_analyzer()
            correlations = analyzer.analyze_correlations(min_samples=1)
            
            if not correlations:
                return "No mood correlations found (need more samples)"
            
            lines = ["Mood Correlations:", "=" * 80]
            
            for mood, entities in correlations.items():
                lines.append(f"\nMood: {mood.upper()}")
                
                sorted_entities = sorted(
                    entities,
                    key=lambda x: x.get('confidence', 0),
                    reverse=True
                )
                
                for entity_info in sorted_entities[:5]:  # Top 5 per mood
                    entity = entity_info['entity']
                    confidence = entity_info['confidence']
                    count = entity_info['count']
                    lines.append(
                        f"  → {entity:40s} ({count:2d} samples, {confidence:.0%} confidence)"
                    )
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Failed to get mood correlations: {e}")
            return f"Error: {e}"
    
    @staticmethod
    def show_mood_stats() -> str:
        """Show overall mood analysis statistics."""
        try:
            from ...rag import get_mood_analyzer
            
            analyzer = get_mood_analyzer()
            
            lines = ["Mood Analysis Statistics:", "=" * 60]
            
            # Get correlation count
            correlations = analyzer.analyze_correlations(min_samples=1)
            total_correlations = sum(len(v) for v in correlations.values())
            
            lines.append(f"Total moods tracked: {len(correlations)}")
            lines.append(f"Total correlations: {total_correlations}")
            
            if correlations:
                avg_correlations = total_correlations / len(correlations)
                lines.append(f"Avg entities per mood: {avg_correlations:.1f}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Failed to get mood stats: {e}")
            return f"Error: {e}"
