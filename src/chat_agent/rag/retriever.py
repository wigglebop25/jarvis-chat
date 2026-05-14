"""RAG retriever that augments LLM context based on query intent."""

import logging
from typing import Any, Optional

from .vector_store import get_vector_store

logger = logging.getLogger(__name__)


class RagRetriever:
    """Retrieves relevant cached data to augment LLM prompts."""
    
    def __init__(self, vector_store: Optional[Any] = None):
        """
        Initialize the RAG retriever.
        
        Args:
            vector_store: Vector store instance (falls back to singleton if None)
        """
        self.vector_store = vector_store or get_vector_store()
    
    def retrieve_context(self, query: str, intent: Optional[str] = None, top_k: int = 3) -> dict[str, Any]:
        """
        Retrieve context from cache based on query and (optionally) detected intent.
        
        Args:
            query: User query text
            intent: Detected intent (MUSIC_CONTROL, PLAYLIST_QUERY, GENERAL_QUERY, etc.)
            top_k: Number of results per query type
            
        Returns:
            Dictionary with retrieved context organized by category
        """
        context: dict[str, Any] = {
            'playlists': [],
            'tracks': [],
            'genres': [],
            'user_profile': None,
            'recent_plays': [],
            'mood_tags': [],
            'similar_queries': [],
        }
        
        try:
            # Based on intent, decide what to retrieve
            if intent == 'MUSIC_CONTROL':
                # For music control: get recent plays and top tracks
                context['recent_plays'] = self.vector_store.get_by_type('recent_play')[:5]
                context['tracks'] = self.vector_store.semantic_search(query, entity_type='track', top_k=top_k)
                context['playlists'] = self.vector_store.semantic_search(query, entity_type='playlist', top_k=top_k)
                
            elif intent == 'PLAYLIST_QUERY':
                # For playlist queries: get playlists and mood tags
                context['playlists'] = self.vector_store.semantic_search(query, entity_type='playlist', top_k=top_k)
                context['mood_tags'] = self.vector_store.get_by_type('mood_tag')[:5]
                context['genres'] = self.vector_store.get_by_type('genre')[:5]
                
            elif intent == 'GENERAL_QUERY':
                # For general queries: get user profile and all context
                context['user_profile'] = self.vector_store.get_by_type('user_profile')
                context['genres'] = self.vector_store.get_by_type('genre')[:3]
                context['mood_tags'] = self.vector_store.get_by_type('mood_tag')[:3]
                
            else:
                # Default: broad semantic search across all
                context['playlists'] = self.vector_store.semantic_search(query, entity_type='playlist', top_k=top_k)
                context['tracks'] = self.vector_store.semantic_search(query, entity_type='track', top_k=top_k)
                context['recent_plays'] = self.vector_store.get_by_type('recent_play')[:3]
            
            # Always include similar past queries for reference
            context['similar_queries'] = self.vector_store.semantic_search(query, entity_type='user_query', top_k=2)
            
        except Exception as e:
            logger.error(f"Failed to retrieve RAG context: {e}")
        
        return context
    
    def format_context_for_prompt(self, context: dict[str, Any]) -> str:
        """
        Format retrieved context into a string to inject into LLM system prompt.
        
        Args:
            context: Context dictionary from retrieve_context()
            
        Returns:
            Formatted context string for inclusion in system prompt
        """
        lines = []
        
        # User profile section
        if context.get('user_profile'):
            profile = context['user_profile'][0] if isinstance(context['user_profile'], list) else context['user_profile']
            lines.append(f"[USER PROFILE]\n{profile.get('entity_name', 'Unknown User')}")
            if profile.get('metadata'):
                lines.append(f"Tier: {profile['metadata'].get('product', 'unknown')}")
        
        # Recent plays section
        if context.get('recent_plays'):
            lines.append("\n[RECENT PLAYS]")
            for track in context['recent_plays'][:3]:
                name = track.get('entity_name', 'Unknown')
                spotify_id = track.get('metadata', {}).get('spotify_id') if isinstance(track.get('metadata'), dict) else None
                suffix = f" [id: {spotify_id}]" if spotify_id else ""
                lines.append(f"• {name}{suffix}")

        if context.get('tracks'):
            lines.append("\n[RELEVANT TRACKS]")
            for track in context['tracks'][:3]:
                name = track.get('entity_name', 'Unknown')
                score = track.get('similarity_score', 0)
                spotify_id = track.get('metadata', {}).get('spotify_id') if isinstance(track.get('metadata'), dict) else None
                artist = track.get('metadata', {}).get('artist', '') if isinstance(track.get('metadata'), dict) else ""
                suffix = f" — {artist}" if artist else ""
                if spotify_id:
                    suffix += f" [id: {spotify_id}]"
                lines.append(f"• {name}{suffix} (relevance: {score:.1%})")

        # Top playlists section
        if context.get('playlists'):
            lines.append("\n[RELEVANT PLAYLISTS]")
            for pl in context['playlists'][:3]:
                name = pl.get('entity_name', 'Unknown')
                score = pl.get('similarity_score', 0)
                spotify_id = pl.get('metadata', {}).get('spotify_id') if isinstance(pl.get('metadata'), dict) else None
                suffix = f" [id: {spotify_id}]" if spotify_id else ""
                lines.append(f"• {name}{suffix} (relevance: {score:.1%})")
        
        # Mood tags section
        if context.get('mood_tags'):
            lines.append("\n[USER MOOD PATTERNS]")
            for tag in context['mood_tags'][:3]:
                keyword = tag.get('entity_name', 'unknown')
                lines.append(f"• {keyword}: {tag.get('text_content', 'pattern found')}")
        
        # Top genres section
        if context.get('genres'):
            lines.append("\n[TOP GENRES]")
            genres = [g.get('entity_name', 'unknown') for g in context['genres'][:5]]
            lines.append(", ".join(genres))
        
        return "\n".join(lines) if lines else ""
    
    def cache_playlist(
        self,
        playlist_id: str,
        playlist_name: str,
        description: str = "",
        ttl_hours: int = 1,
    ) -> bool:
        """Cache a Spotify playlist for semantic search."""
        text_content = f"{playlist_name} {description}".strip()
        metadata = {'spotify_id': playlist_id}
        return self.vector_store.embed_and_cache(
            entity_id=f"playlist:{playlist_id}",
            entity_type='playlist',
            text_content=text_content,
            entity_name=playlist_name,
            metadata=metadata,
            ttl_hours=ttl_hours,
        )
    
    def cache_track(
        self,
        track_id: str,
        track_name: str,
        artist: str = "",
        ttl_hours: int = 6,
    ) -> bool:
        """Cache a Spotify track for semantic search."""
        text_content = f"{track_name} by {artist}".strip()
        metadata = {'spotify_id': track_id, 'artist': artist}
        return self.vector_store.embed_and_cache(
            entity_id=f"track:{track_id}",
            entity_type='track',
            text_content=text_content,
            entity_name=track_name,
            metadata=metadata,
            ttl_hours=ttl_hours,
        )
    
    def cache_user_profile(
        self,
        user_id: str,
        display_name: str,
        top_genres: list[str],
        product: str = "free",
    ) -> bool:
        """Cache user profile info."""
        text_content = f"{display_name} likes {' '.join(top_genres)}"
        metadata = {'spotify_id': user_id, 'top_genres': top_genres, 'product': product}
        return self.vector_store.embed_and_cache(
            entity_id=f"user:{user_id}",
            entity_type='user_profile',
            text_content=text_content,
            entity_name=display_name,
            metadata=metadata,
            ttl_hours=24,
        )
    
    def cache_mood_correlation(
        self,
        mood_keyword: str,
        playlist_id: str,
        playlist_name: str,
        confidence: float = 0.8,
    ) -> bool:
        """
        Cache a mood-to-playlist correlation.
        
        Example: mood_keyword="sad", playlist="for when i want to cry"
        """
        text_content = f"When user says '{mood_keyword}', they often listen to '{playlist_name}'"
        metadata = {'mood': mood_keyword, 'playlist_id': playlist_id, 'confidence': confidence}
        return self.vector_store.embed_and_cache(
            entity_id=f"mood:{mood_keyword}:{playlist_id}",
            entity_type='mood_tag',
            text_content=text_content,
            entity_name=f"{mood_keyword} → {playlist_name}",
            metadata=metadata,
            ttl_hours=24 * 7,  # Keep mood correlations longer
        )


# Singleton instance
_rag_retriever = None


def get_rag_retriever() -> RagRetriever:
    """Get or create the singleton RAG retriever."""
    global _rag_retriever
    if _rag_retriever is None:
        _rag_retriever = RagRetriever()
    return _rag_retriever
