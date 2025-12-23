from typing import Dict, Any, List
from src.core.tools import Tool, ToolParameter
from src.memory.memory_retriever import MemoryRetriever
from src.utils.embeddings import EmbeddingService
from src.utils.memory_logger import memory_logger

class MemorySearchTool:
    """Tool for active memory search that agents can call explicitly."""

    def __init__(self, retriever: MemoryRetriever, embedding_service: EmbeddingService):
        self.retriever = retriever
        self.embedding_service = embedding_service

    def search_memory(
        self,
        query: str,
        search_type: str = "both",
        limit: int = 5
    ) -> str:
        """
        Search through episodic and/or semantic memories.

        Args:
            query: The search query
            search_type: What type of memory to search ('episodic', 'semantic', or 'both')
            limit: Maximum results to return per memory type

        Returns:
            Formatted search results
        """
        try:
            memory_logger.log_event("tool_memory_search", {
                "query": query,
                "search_type": search_type,
                "limit": limit
            })

            # Generate embedding for query
            query_embedding = self.embedding_service.embed(query)

            # Search memories
            results = {}
            if search_type in ['episodic', 'both']:
                results['episodic'] = self.retriever.search_memories(
                    query_embedding=query_embedding,
                    search_type='episodic',
                    limit=limit,
                    filters={'status': 'active', 'confidence__gt': 0.5}
                )

            if search_type in ['semantic', 'both']:
                results['semantic'] = self.retriever.search_memories(
                    query_embedding=query_embedding,
                    search_type='semantic',
                    limit=limit,
                    filters={}
                )

            # Format results
            formatted_results = [f"Memory search results for '{query}':\n"]

            # Episodic memories
            if 'episodic' in results and results['episodic']:
                formatted_results.append("**Episodic Memories:**")
                for i, mem in enumerate(results['episodic'], 1):
                    summary = mem.get('summary', 'No summary')
                    timestamp = mem.get('timestamp', '')
                    importance = mem.get('importance', 0.0)
                    similarity = mem.get('similarity', 0.0)

                    formatted_results.append(f"{i}. [{timestamp[:10]}] {summary}")
                    formatted_results.append(f"   Importance: {importance:.2f}, Similarity: {similarity:.2f}")
                    formatted_results.append("")

            # Semantic facts
            if 'semantic' in results and results['semantic']:
                formatted_results.append("**Semantic Facts:**")
                for i, fact in enumerate(results['semantic'], 1):
                    fact_text = fact.get('fact', 'No fact')
                    confidence = fact.get('confidence', 0.0)
                    similarity = fact.get('similarity', 0.0)

                    formatted_results.append(f"{i}. {fact_text}")
                    formatted_results.append(f"   Confidence: {confidence:.2f}, Similarity: {similarity:.2f}")
                    formatted_results.append("")

            if not any(results.values()):
                formatted_results.append("No memories found matching the query.")

            result_text = "\n".join(formatted_results)

            memory_logger.log_event("tool_memory_search_complete", {
                "episodic_count": len(results.get('episodic', [])),
                "semantic_count": len(results.get('semantic', []))
            })

            return result_text

        except Exception as e:
            error_msg = f"Memory search failed: {str(e)}"
            memory_logger.log_event("tool_memory_search_error", {"error": str(e)})
            return error_msg

# Create tool instance (will be initialized with dependencies)
def create_memory_search_tool(retriever: MemoryRetriever, embedding_service: EmbeddingService) -> Tool:
    """Factory function to create the memory search tool with dependencies."""
    tool_instance = MemorySearchTool(retriever, embedding_service)

    return Tool(
        name="search_memory",
        description="Search through episodic memories (past conversations) and semantic facts. Use this when you need to recall specific information from previous interactions or established facts.",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="The search query - describe what you want to find in memory",
                required=True
            ),
            ToolParameter(
                name="search_type",
                type="string",
                description="Type of memory to search: 'episodic' (past conversations), 'semantic' (facts), or 'both'",
                required=False,
                enum=["episodic", "semantic", "both"]
            ),
            ToolParameter(
                name="limit",
                type="number",
                description="Maximum results to return per memory type (1-10)",
                required=False
            )
        ],
        function=tool_instance.search_memory
    )
