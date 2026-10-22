import os
import time
import uuid
from typing import Any, Literal

import msgpack
from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_mongodb.agent_toolkit import MONGODB_AGENT_SYSTEM_PROMPT
from langchain_mongodb.agent_toolkit.database import MongoDBDatabase
from langchain_mongodb.agent_toolkit.toolkit import MongoDBDatabaseToolkit
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr
from pymongo import MongoClient

load_dotenv()


mdl = 'nvidia/Qwen3.8-27B-NVFP4'

embeddings = OllamaEmbeddings(model='qwen3-embedding')

llm = ChatOpenAI(
    model=mdl,
    temperature=0.5,
    max_completion_tokens=4096,
    streaming=True,
    # reasoning=False,
    base_url='http://localhost:8000/v1',
    extra_body={"top_k": 20},
    top_p=0.95,
    # presence_penalty=0,
    # min_p=0,
    api_key=SecretStr('not-a-real-key'),
)
base_uri = "mongodb://localhost:27017/"
username = "unia-search-ai"
password = os.getenv('MONGO_DB_UNIA_SEARCH_AI_PASSWORD')
auth_source = "unia"

# Construct the full URI
connection_uri = f"mongodb://{username}:{password}@localhost:27017/?authSource={auth_source}"


db = MongoDBDatabase.from_connection_string(
    connection_uri, database="unia"
)

client = MongoClient(connection_uri, appname='intrabot_text_to_mql_agent')

# Create toolkit and extract tools
toolkit = MongoDBDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
tool = {t.name: t for t in tools}

print("🛠️ Available Tools:", list(tool.keys()))



class LLMSummarizingMongoDBSaver(MongoDBSaver):
    """MongoDB saver with LLM-powered intelligent summarization"""

    def __init__(self, client, llm):
        super().__init__(client)
        self.llm = llm

        # Cache for performance (optional)
        self._summary_cache = {}

    def summarize_step(self, checkpoint_data: dict[str, Any]) -> str:
        """Generate contextual summary using LLM"""
        try:
            # Extract channel values and messages
            channel_values = checkpoint_data.get("channel_values", {})
            messages = channel_values.get("messages", [])

            if not messages:
                return "🔄 Initial state"

            # Get the most recent message
            last_message = messages[-1]

            if not last_message:
                return "📭 Empty step"

            # Extract message details
            message_type = (
                type(last_message).__name__
                if hasattr(last_message, "__class__")
                else "unknown"
            )
            content = getattr(last_message, "content", "") or ""
            tool_calls = getattr(last_message, "tool_calls", [])

            # Handle dict-like messages (fallback)
            if isinstance(last_message, dict):
                message_type = last_message.get("type", "unknown")
                content = last_message.get("content", "")
                tool_calls = last_message.get("tool_calls", [])

            # Create a simple cache key to avoid redundant LLM calls
            cache_key = f"{message_type}:{content[:50]}:{len(tool_calls)}"
            if cache_key in self._summary_cache:
                return self._summary_cache[cache_key]

            # Build context for LLM
            context_parts = []
            if content:
                context_parts.append(f"Content: {content[:200]}")
            if tool_calls:
                tool_info = []
                for tc in tool_calls[:2]:  # Limit to first 2 tool calls
                    tool_name = tc.get("name", "unknown")
                    tool_args = str(tc.get("args", {}))[:100]
                    tool_info.append(f"{tool_name}({tool_args})")
                context_parts.append(f"Tool calls: {', '.join(tool_info)}")

            context = "\n".join(context_parts) if context_parts else "No content"

            # LLM prompt for summarization
            prompt = f"""Summarize this conversation step in 2-5 words with a relevant emoji.

Message type: {message_type}
{context}

Guidelines:
- Use emojis: 👤 for user, 🤖 for AI, 🔧 for tools, 📊 for data, ✨ for results
- Be concise and descriptive
- Focus on the action/intent

Examples:
- "👤 Count movies query"
- "🔧 Schema lookup: collection"
- "📊 Aggregation pipeline"
- "✨ Formatted results"
- "❌ Query validation error"

Summary:"""

            # Get LLM response
            response = self.llm.invoke(prompt)
            summary = response.content.strip()[:60]  # Limit length

            # Cache the result
            self._summary_cache[cache_key] = summary

            # Keep cache size reasonable
            if len(self._summary_cache) > 100:
                # Remove oldest entries (simple FIFO)
                oldest_keys = list(self._summary_cache.keys())[:50]
                for key in oldest_keys:
                    del self._summary_cache[key]

            return summary

        except Exception as e:
            # Fallback for any errors
            error_msg = str(e)[:30]
            return f"❓ Step (error: {error_msg}...)"

    def put(
        self,
        config: RunnableConfig,
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any],
    ) -> RunnableConfig:
        """Override put method to add LLM-generated step summary"""
        try:
            # Generate step summary using LLM
            step_summary = self.summarize_step(checkpoint)

            # Create enhanced metadata
            enhanced_metadata = metadata.copy() if metadata else {}
            enhanced_metadata["step_summary"] = step_summary
            enhanced_metadata["step_timestamp"] = checkpoint.get("ts", "unknown")

            # Add step number if available
            messages = checkpoint.get("channel_values", {}).get("messages", [])
            enhanced_metadata["step_number"] = len(messages)

            # Call parent's put method
            return super().put(config, checkpoint, enhanced_metadata, new_versions)

        except Exception as e:
            print(f"❌ Error adding LLM summary: {e}")
            # Fallback to basic metadata
            return super().put(config, checkpoint, metadata, new_versions)


def inspect_thread_with_summaries_enhanced(
    thread_id: str, limit: int = 20, show_details: bool = False
):
    """Enhanced thread inspection with better formatting"""
    try:
        db_checkpoints = client["checkpointing_db"]
        collection = db_checkpoints.checkpoints

        # Get checkpoints for this thread
        checkpoints = list(
            collection.find({"thread_id": thread_id}).sort("_id", 1).limit(limit)
        )

        if not checkpoints:
            print(f"❌ No checkpoints found for thread: {thread_id}")
            return []

        print(f"\n🔍 Thread History: {thread_id}")
        print(f"📊 Total steps: {len(checkpoints)}")
        print("=" * 80)

        # Group consecutive similar operations
        last_summary = None
        consecutive_count = 0

        for i, checkpoint_doc in enumerate(checkpoints, 1):
            # Get timestamp
            timestamp = checkpoint_doc["_id"].generation_time
            time_str = timestamp.strftime("%H:%M:%S")

            # Get metadata
            metadata = checkpoint_doc.get("metadata", {})

            # Handle both binary and dict formats
            if isinstance(metadata, dict):
                step_summary = metadata.get("step_summary", "No summary")
            else:
                try:
                    decoded_metadata = msgpack.unpackb(
                        metadata, raw=False, strict_map_key=False
                    )
                    step_summary = decoded_metadata.get("step_summary", "No summary")
                except (msgpack.UnpackException, ValueError) as e:
                    step_summary = "Unable to decode"

            # Clean up display
            if isinstance(step_summary, bytes):
                step_summary = step_summary.decode("utf-8", errors="replace")

            # Group similar consecutive operations
            if step_summary == last_summary and not show_details:
                consecutive_count += 1
            else:
                if consecutive_count > 0:
                    print(f"   └─ (repeated {consecutive_count} more times)")

                print(f"\n📍 Step {i} [{time_str}]")
                print(f"   {step_summary}")

                last_summary = step_summary
                consecutive_count = 0

        if consecutive_count > 0:
            print(f"   └─ (repeated {consecutive_count} more times)")

        print("\n" + "=" * 80)
        return checkpoints

    except Exception as e:
        print(f"❌ Error inspecting thread: {e}")
        import traceback

        traceback.print_exc()
        return []


print("🔄 UPDATING AGENTS WITH LLM-POWERED SUMMARIZATION")
print("=" * 60)

FORMAT_PROMPT = (
    "Answer the user's question using the MongoDB query result (JSON) below. "
    "Use Markdown (tables or lists where fitting). Do not show raw JSON or _id values. "
    "If the result is empty, say that no matching documents were found."
)

def temp_create_final_answer(state):
    """Enhanced format function with large dataset handling"""

    question = next(m.content for m in reversed(state["messages"]) if m.type == "human")
    result = state["messages"][-1].content
    resp = llm.invoke([
        SystemMessage(content=FORMAT_PROMPT),
        HumanMessage(content=f"Question: {question}\n\nQuery result:\n{result[:20000]}"),
    ])
    return {"messages": [resp]}


# Tool nodes for LangGraph
schema_node = ToolNode([tool["mongodb_schema"]], name="get_schema")
run_node = ToolNode([tool["mongodb_query"]], name="run_query")

def list_collections(state: MessagesState):
    """Deterministic node to list available collections"""
    call = {
        "name": "mongodb_list_collections",
        "args": {},
        "id": "abc",
        "type": "tool_call",
    }
    call_msg = AIMessage(content="", tool_calls=[call])
    resp = tool["mongodb_list_collections"].invoke(call)
    summary = AIMessage(f"Available collections: {resp.content}")
    return {"messages": [call_msg, resp, summary]}

def call_get_schema(state: MessagesState):
    """Prompt LLM to select and call schema tool"""
    llm_with = llm.bind_tools([tool["mongodb_schema"]], tool_choice="any")
    resp = llm_with.invoke(state["messages"])
    return {"messages": [resp]}


def generate_query(state: MessagesState):
    """Generate MongoDB aggregation pipeline"""

    projection_prompt = '''Schließe Felder, die du nicht brauchen wirst, direkt aus. Limitiere die Anzahl der zurückgegebenen Suchergebnisse IMMER streng. NUR wenn du nach einem einzigen Element mittels der _id suchst, darfst du alle Felder zurückgeben.
mhbs: Füge bei Modulhandbüchern in der Collection mhbs IMMER eine Projektion hinzu, die NUR folgende Felder oder eine Auswahl folgender Felder enthält: "_hash", "path", "name", "start_semester", "faculties", "mhb_group", "description", "module_groups". Bei "modules_groups" darfst du, falls nötig ein lookup auf modules machen, dann darfst du allerdings von modules NUR die Felder "_id", "module_code", "name" auswählen mit einer Projektion.
modules: Füge bei Modulen in der Collection modules IMMER eine Projektion hinzu, die IMMER mindestens folgende Felder oder noch weitere Felder ausschließt: "_hash", "inhalt", "ziele", "workloads", "klausurbeschreibung", "voraussetzungen", "bestehensvoraussetzungen", "dozent", "international", "prüfungen".
exams: Bei Klausuren muss nur "_hash" ausgeschlossen werden, falls du bestimmte Felder aber nicht brauchst, solltest du sie dennoch ausschließen.'''

    llm_with = llm.bind_tools([tool["mongodb_query"]])
    resp = llm_with.invoke(
        [{"role": "system", "content": MONGODB_AGENT_SYSTEM_PROMPT + '\n\n' + projection_prompt}] + state["messages"]
    )
    return {"messages": [resp]}


def check_query(state: MessagesState):
    """Validate and sanitize generated query"""
    original = state["messages"][-1].tool_calls[0]["args"]["query"]
    resp = llm.bind_tools([tool["mongodb_query"]], tool_choice="any").invoke(
        [
            {"role": "system", "content": MONGODB_AGENT_SYSTEM_PROMPT + '\n\nReview this MongoDB query. If it is correct, call mongodb_query with EXACTLY the same query. Otherwise call it with a corrected version. Never write admin or schema commands like getCollectionNames.'},
            {"role": "user", "content": original},
        ]
    )
    resp.id = state["messages"][-1].id
    return {"messages": [resp]}


def need_checker(state: MessagesState) -> Literal[END, "check_query"]:
    """Conditional edge: run checker if tool call present"""
    return "check_query" if state["messages"][-1].tool_calls else END



def create_langgraph_agent_with_enhanced_memory():
    """Create custom LangGraph agent with LLM-powered summarizing checkpointer"""
    summarizing_checkpointer = LLMSummarizingMongoDBSaver(client, llm)

    # Build the graph
    g = StateGraph(MessagesState)

    # Add nodes
    g.add_node("list_collections", list_collections)
    g.add_node("call_get_schema", call_get_schema)
    g.add_node("get_schema", schema_node)
    g.add_node("generate_query", generate_query)
    g.add_node("check_query", check_query)
    g.add_node("run_query", run_node)
    g.add_node("temp_create_final_answer", temp_create_final_answer)

    # Add edges - temp_create_final_answer goes directly to END
    g.add_edge(START, "list_collections")
    g.add_edge("list_collections", "call_get_schema")
    g.add_edge("call_get_schema", "get_schema")
    g.add_edge("get_schema", "generate_query")
    g.add_conditional_edges("generate_query", need_checker)
    g.add_edge("check_query", "run_query")
    g.add_edge("run_query", "temp_create_final_answer")
    g.add_edge("temp_create_final_answer", END)  # Direct to END - checkpoints handle persistence

    return g.compile(checkpointer=summarizing_checkpointer)

# Create the enhanced agents
mongo_agent_with_memory = create_langgraph_agent_with_enhanced_memory()

print("✅ Agents created with LLM-powered summarization!")
print("\n📖 Features:")
print("• Works with any MongoDB database and collection")
print("• Uses LLM to intelligently summarize each step")
print("• Adapts to any query type automatically")
print("• Provides natural language step descriptions")
print("• Caches summaries for better performance")


def execute_graph_with_memory(thread_id: str, user_input: str):
    """Execute LangGraph agent with persistent memory"""
    config = {"configurable": {"thread_id": thread_id}}

    print(f"🧵 Thread: {thread_id}")
    print(f"❓ Query: {user_input}")
    print("📊 Agent: Custom LangGraph")
    print("=" * 50)

    for step in mongo_agent_with_memory.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()



def list_conversation_threads():
    """List all available conversation threads"""
    try:
        # Check the main checkpoint database used by our agents
        db_checkpoints = client["checkpointing_db"]
        collection = db_checkpoints.checkpoints

        threads = collection.distinct("thread_id")
        total_checkpoints = collection.count_documents({})

        print("📋 Available Conversation Threads:")
        print(f"📊 Total checkpoints: {total_checkpoints}")
        print("=" * 50)

        for i, thread_id in enumerate(threads, 1):
            count = collection.count_documents({"thread_id": thread_id})
            print(f"  {i}. Thread: {thread_id}")
            print(f"     └─ {count} checkpoints")

        return threads

    except Exception as e:
        print(f"❌ Error listing threads: {e}")
        return []

def inspect_thread_history(thread_id: str, limit: int = 10):
    """Inspect conversation history for a specific thread"""
    try:
        # Use the enhanced inspection function if available
        return inspect_thread_with_summaries_enhanced(thread_id, limit)
    except NameError:
        # Fallback to basic inspection
        try:
            db_checkpoints = client["checkpointing_db"]
            collection = db_checkpoints.checkpoints

            checkpoints = list(
                collection.find({"thread_id": thread_id})
                .sort("checkpoint_ns", -1)
                .limit(limit)
            )

            if not checkpoints:
                print(f"❌ No checkpoints found for thread: {thread_id}")
                return []

            print(f"🔍 Thread History: {thread_id}")
            print(f"📊 Showing {len(checkpoints)} most recent checkpoints")
            print("=" * 60)

            for i, checkpoint in enumerate(reversed(checkpoints), 1):
                print(f"\n📍 Step {i}:")

                channel_values = checkpoint.get("channel_values", {})
                if "messages" in channel_values:
                    messages = channel_values["messages"]
                    print(f"   Messages: {len(messages)} total")

                    if messages:
                        last_msg = messages[-1]
                        if isinstance(last_msg, dict):
                            content = last_msg.get("content", "")
                            tool_calls = last_msg.get("tool_calls", [])

                            if tool_calls:
                                tool_name = tool_calls[0].get("name", "unknown")
                                print(f"   🔧 Tool Call: {tool_name}")
                            elif content:
                                preview = (
                                    content[:100] + "..."
                                    if len(content) > 100
                                    else content
                                )
                                print(f"   💬 Content: {preview}")

            return checkpoints

        except Exception as e:
            print(f"❌ Error inspecting thread: {e}")
            return []


def clear_thread_history(thread_id: str):
    """Clear conversation history for a specific thread"""
    try:
        db_checkpoints = client["checkpointing_db"]

        # Clear main checkpoints
        collection = db_checkpoints.checkpoints
        result = collection.delete_many({"thread_id": thread_id})
        print(f"🗑️ Cleared {result.deleted_count} checkpoints from thread: {thread_id}")

        # Clear checkpoint writes
        writes_collection = db_checkpoints.checkpoint_writes
        writes_result = writes_collection.delete_many({"thread_id": thread_id})
        print(f"🗑️ Cleared {writes_result.deleted_count} checkpoint writes")

    except Exception as e:
        print(f"❌ Error clearing thread: {e}")

def memory_system_stats():
    """Show comprehensive memory statistics"""
    try:
        db_checkpoints = client["checkpointing_db"]
        checkpoints = db_checkpoints.checkpoints
        checkpoint_writes = db_checkpoints.checkpoint_writes

        total_checkpoints = checkpoints.count_documents({})
        total_writes = checkpoint_writes.count_documents({})
        total_threads = len(checkpoints.distinct("thread_id"))

        print("📊 Memory System Statistics")
        print("=" * 40)
        print(f"💾 Total checkpoints: {total_checkpoints}")
        print(f"✍️ Total checkpoint writes: {total_writes}")
        print(f"🧵 Total conversation threads: {total_threads}")
        print("🏛️ Database: checkpointing_db")

        return {
            "checkpoints": total_checkpoints,
            "writes": total_writes,
            "threads": total_threads,
        }

    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return {}


################################################################

def demo_basic_queries():
    """Demonstrate basic text-to-MQL functionality"""
    print("🎬 DEMO: Basic Text-to-MQL Queries")
    print("=" * 50)

    queries = [
        "Finde das Modul mit der Kennung GES-0001",
        "Welche Module hat der Studiengang Data Science",
        "Welche Klausuren muss ich im Studiengang Data Science in der Mathegruppe absolvieren",
    ]

    for i, query in enumerate(queries, 1):
        thread_id = f"demo_basic_{i}"
        print(f"\n--- Demo Query {i} ---")
        print(f"Query: {query}")
        print()

        execute_graph_with_memory(thread_id, query)

        if i < len(queries):
            print("\n" + "=" * 50)

'''

def demo_conversation_memory():
    """Demonstrate conversation memory across multiple related queries"""
    thread_id = f"conversation_demo_{uuid.uuid4().hex[:8]}"

    print("🎬 DEMO: Conversation Memory with Text-to-MQL")
    print("=" * 50)

    conversation = [
        "List the top 3 directors by movie count",
        "What was the movie count for the first director?",
        "Show me movies by that director with highest ratings",
    ]

    for i, query in enumerate(conversation, 1):
        print(f"\n--- Conversation Step {i} ---")
        execute_graph_with_memory(thread_id, query)

        if i < len(conversation):
            print("\n🔄 Building context for next query...")
            print("=" * 40)

    print("\n🔍 Complete Conversation Analysis:")
    print("=" * 40)
    inspect_thread_history(thread_id)

def test_memory_functionality():
    """Test memory functionality with a simple example"""
    thread_id = f"memory_test_{uuid.uuid4().hex[:8]}"

    print("🧪 TESTING: Memory Functionality")
    print("=" * 50)

    print("Step 1: Ask about directors")
    execute_graph_with_memory(thread_id, "List top 3 directors by movie count")

    print("\nStep 2: Follow up question (tests memory)")
    execute_graph_with_memory(
        thread_id, "What was the movie count for the first director?"
    )

    print("\n🔍 Memory Analysis:")
    inspect_thread_history(thread_id)

    return thread_id


def test_enhanced_summarization():
    """Test the enhanced summarization system with various query patterns"""
    print("\n🧪 TESTING ENHANCED SUMMARIZATION SYSTEM")
    print("=" * 60)

    thread_id = f"enhanced_test_{uuid.uuid4().hex[:8]}"

    # Test various query patterns
    test_queries = [
        "How many movies are in the database?",
        "Find the average rating of all movies",
        "Show me the top 5 directors by movie count",
    ]

    print(f"Testing thread: {thread_id}")
    print("Running query patterns with enhanced summarization...")
    print("=" * 50)

    for i, query in enumerate(test_queries, 1):
        print(f"\n📌 Test {i}: {query}")
        execute_graph_with_memory(thread_id, query)
        print(f"✅ Test {i} complete")

    # Inspect the results with enhanced summaries
    print("\n🔍 Enhanced Thread Analysis:")
    print("=" * 50)
    inspect_thread_history(thread_id)

    return thread_id


def compare_agents_with_memory(
    query: str, max_retries: int = 3, recursion_limit: int = 50
):
    """
    Side-by-side comparison of ReAct vs LangGraph agents using a specified query.

    Parameters:
    -----------
    query : str
        The natural language query to test with both agents
    max_retries : int, default=3
        Maximum number of retry attempts if an agent fails
    recursion_limit : int, default=50
        Maximum recursion depth for the ReAct agent to prevent infinite loops

    Comparison points:
    -----------------
    - Execution style: ReAct's autonomous reasoning vs LangGraph's structured workflow
    - Memory patterns: How each agent stores conversation state
    - Output format: Differences in result presentation
    - Error handling: How each agent responds to failures
    """
    base_thread = f"compare_{uuid.uuid4().hex[:8]}"

    print("Agent Comparison: LangGraph")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Max Retries: {max_retries}")
    print(f"Recursion Limit: {recursion_limit}")
    print("=" * 60)

    graph_results = {
        "success": False,
        "attempts": 0,
        "error": None,
        "execution_time": None,
    }

    # Test LangGraph Agent
    print("\nLangGraph Agent Execution:")
    print("-" * 40)

    start_time = time.time()

    for attempt in range(max_retries):
        graph_results["attempts"] = attempt + 1
        thread_id = f"{base_thread}_graph_attempt_{attempt + 1}"

        print(f"\nAttempt {attempt + 1}/{max_retries}")
        print(f"Thread: {thread_id}")

        try:
            config = {"configurable": {"thread_id": thread_id}}

            step_count = 0
            print("Execution steps:")
            for step in mongo_agent_with_memory.stream(
                {"messages": [{"role": "user", "content": query}]},
                config,
                stream_mode="values",
            ):
                step_count += 1
                last_msg = step["messages"][-1]

                # Show step summary
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tool_name = last_msg.tool_calls[0]["name"]
                    print(f"  Step {step_count}: Tool call: {tool_name}")
                elif hasattr(last_msg, "content") and last_msg.content:
                    content_preview = last_msg.content[:50] + (
                        "..." if len(last_msg.content) > 50 else ""
                    )
                    print(f"  Step {step_count}: Response: {content_preview}")

                # Show full output for final step
                if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
                    print("\nFinal LangGraph Response:")
                    last_msg.pretty_print()

            graph_results["success"] = True
            graph_results["execution_time"] = time.time() - start_time
            print(f"\nLangGraph agent succeeded in {step_count} steps")
            break

        except Exception as e:
            graph_results["error"] = str(e)
            print(f"\nLangGraph attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                print("Retrying with fresh thread...")
            else:
                print("Max retries reached for LangGraph agent")
                graph_results["execution_time"] = time.time() - start_time

    # Comparison Summary
    print("\nComparison Summary:")
    print("=" * 60)

    print("\nLangGraph Agent Results:")
    print(f"  Success: {'✅' if graph_results['success'] else '❌'}")
    print(f"  Attempts: {graph_results['attempts']}/{max_retries}")
    print(
        f"  Execution Time: {graph_results['execution_time']:.2f}s"
        if graph_results["execution_time"]
        else "  Execution Time: N/A"
    )
    if graph_results["error"]:
        print(f"  Final Error: {graph_results['error']}")

    print("  LangGraph Agent:")
    print("    - Structured, deterministic workflow")
    print("    - Predefined step sequence with conditional branches")
    print("    - Better error isolation and recovery")
    print("    - More predictable but less flexible execution")

    if graph_results["success"]:
        print("\nMemory Pattern Analysis:")
        print("  LangGraph Agent Memory:")
        graph_thread = f"{base_thread}_graph_attempt_{graph_results['attempts']}"
        try:
            inspect_thread_history(graph_thread, limit=3)
        except Exception as e:
            print("Unable to inspect LangGraph memory")

    return {
        "langgraph": graph_results,
        "query": query,
        "config": {"max_retries": max_retries, "recursion_limit": recursion_limit},
    }




def test_simple_comparison():
    """Test with a simple query that should work"""
    simple_query = "Count the total number of movies in the database"
    return compare_agents_with_memory(simple_query, max_retries=2, recursion_limit=30)


def test_moderate_comparison():
    """Test with a moderately complex query"""
    moderate_query = "List the top 5 directors who have directed the most movies"
    return compare_agents_with_memory(moderate_query, max_retries=2, recursion_limit=40)


def test_complex_comparison():
    """Test with the original complex query that caused issues"""
    complex_query = (
        "Find the top 5 directors with most award wins and at least 5 movies"
    )
    return compare_agents_with_memory(complex_query, max_retries=3, recursion_limit=50)


def run_comparison_tests():
    """Run a series of comparison tests with different query complexities"""
    print("Running Comparison Test Suite")
    print("=" * 60)

    tests = [
        ("Simple Query", test_simple_comparison),
        ("Moderate Query", test_moderate_comparison),
        ("Complex Query", test_complex_comparison),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = None

    return results


print("✅ Enhanced agent comparison functions loaded!")
print("\nUsage examples:")
print('compare_agents_with_memory("Count all movies", max_retries=2)')
print(
    'compare_agents_with_memory("Find top directors", max_retries=3, recursion_limit=40)'
)
print("run_comparison_tests()  # Run multiple test scenarios")



def interactive_query():
    """Interactive query interface with memory"""
    print("🔍 Interactive Text-to-MQL Query Interface")
    print(
        "Commands: 'exit' to quit, 'threads' to list, 'switch ' to change thread"
    )
    print("=" * 70)

    thread_id = f"interactive_{uuid.uuid4().hex[:8]}"

    while True:
        try:
            user_input = input(f"\n[{thread_id}] Enter your query: ").strip()

            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "threads":
                list_conversation_threads()
                continue
            elif user_input.lower().startswith("switch "):
                thread_id = user_input[7:].strip()
                print(f"🔄 Switched to thread: {thread_id}")
                continue
            elif user_input.lower() == "debug":
                inspect_thread_history(thread_id)
                continue
            elif not user_input:
                continue

            print()
            execute_graph_with_memory(thread_id, user_input)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
'''


if __name__ == "__main__":
    demo_basic_queries()
