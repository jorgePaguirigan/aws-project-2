"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the step-by-step solution files and INSTRUCTIONS.md for guidance.
Do NOT copy the solution directly — work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── App Initialisation ────────────────────────────────────────────────────────
app = BedrockAgentCoreApp()

# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── Configuration ──────────────────────────────────────────────────────────────
GATEWAY_URL = "https://customersupportgateway-aytlvjww9v.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID       = "JZ6ZLYIMQV"
REGION      = "us-east-1"
MEMORY_ID   = "CustomerSupportMemory-FOLe152QaP"


# ── Model and Clients ─────────────────────────────────────────────────────────
model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ── Namespace Helper ──────────────────────────────────────────────────────────
def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string.

    Reads `namespaceTemplates` (current API field) or falls back to the
    legacy `namespaces` field, whichever the strategy dict provides.
    """
    strategies = mem_client.get_memory_strategies(memory_id)
    namespaces = {}
    for s in strategies:
        templates = s.get("namespaceTemplates") or s.get("namespaces")
        if templates:
            namespaces[s["type"]] = templates[0]
    return namespaces


# ── Memory Hook ────────────────────────────────────────────────────────────────
class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)

    @staticmethod
    def _is_plain_user_text(message: dict):
        """Return the text of a message if it's a plain user text message
        (not a tool result), else None."""
        if message.get("role") != "user":
            return None
        content = message.get("content", [])
        if any("toolResult" in block for block in content):
            return None
        texts = [block["text"] for block in content if "text" in block]
        return texts[0] if texts else None

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        messages = event.agent.messages
        if not messages:
            return
        last_message = messages[-1]

        query = self._is_plain_user_text(last_message)
        if not query:
            return

        memory_lines = []
        for strategy_type, namespace_template in self.namespaces.items():
            namespace = namespace_template.format(actorId=self.actor_id)
            try:
                results = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=query,
                    top_k=5,
                )
            except Exception as e:
                logger.warning(f"Memory retrieve failed for {namespace}: {e}")
                continue

            for r in results:
                text = r.get("content", {}).get("text", "")
                if text:
                    memory_lines.append(f"[{strategy_type}] {text}")

        if memory_lines:
            context_block = "\n".join(memory_lines)
            new_text = f"Customer Context:\n{context_block}\n\n{query}"
            last_message["content"] = [{"text": new_text}]

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        messages = event.agent.messages

        user_query = None
        assistant_response = None

        for msg in reversed(messages):
            if msg.get("role") == "assistant" and assistant_response is None:
                texts = [b["text"] for b in msg.get("content", []) if "text" in b]
                if texts:
                    assistant_response = texts[0]
            elif msg.get("role") == "user" and user_query is None:
                text = self._is_plain_user_text(msg)
                if text:
                    user_query = text
            if user_query and assistant_response:
                break

        if not (user_query and assistant_response):
            return

        try:
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[(user_query, "USER"), (assistant_response, "ASSISTANT")],
            )
        except Exception as e:
            logger.warning(f"Failed to save support interaction to memory: {e}")

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register both memory callbacks."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── Knowledge Base Tool ─────────────────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID:
        return "Knowledge base not configured."

    try:
        resp = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
    except Exception as e:
        logger.error(f"Knowledge base retrieve failed: {e}")
        return f"Knowledge base search failed: {e}"

    results = resp.get("retrievalResults", [])
    if not results:
        return "No relevant information found in the knowledge base."

    chunks = [
        r["content"]["text"]
        for r in results
        if r.get("content", {}).get("text")
    ]
    return "\n---\n".join(chunks)


# ── Loyalty Discount Tool (Code Interpreter) ─────────────────────────────────
@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    # NOTE: point redemption conversion rate (100 points = $1) is an
    # assumption — adjust to match your course's spec if different.
    code = f"""
import json

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

loyalty_points = {loyalty_points}
tier = "{tier}"
order_total = {order_total}
product_category = "{product_category}"

POINTS_PER_DOLLAR = 100  # 100 points redeemed = $1 off

# Floor available points to nearest 500, cap redemption at 50% of order total
floored_points = (loyalty_points // 500) * 500
max_redeemable_value = order_total * 0.5

points_redeemed = 0
points_redeemed_value = 0.0
while floored_points > 0:
    candidate_value = floored_points / POINTS_PER_DOLLAR
    if candidate_value <= max_redeemable_value:
        points_redeemed = floored_points
        points_redeemed_value = candidate_value
        break
    floored_points -= 500

subtotal_after_points = order_total - points_redeemed_value
tier_discount_pct = tier_rates.get(tier, 0.0)
tier_discount = round(subtotal_after_points * tier_discount_pct, 2)
final_total = round(subtotal_after_points - tier_discount, 2)

total_savings = round(points_redeemed_value + tier_discount, 2)
points_earned = int(order_total * earn_rates.get(product_category, 1))
remaining_points = loyalty_points - points_redeemed + points_earned

result = {{
    "points_redeemed": points_redeemed,
    "points_redeemed_value": round(points_redeemed_value, 2),
    "tier_discount_pct": tier_discount_pct,
    "tier_discount": tier_discount,
    "final_total": final_total,
    "total_savings": total_savings,
    "points_earned": points_earned,
    "remaining_points": remaining_points,
}}

print(json.dumps(result))
"""

    try:
        with code_session(REGION) as code_client:
            response = code_client.invoke(
                "executeCode",
                {
                    "code": code,
                    "language": "python",
                    "clearContext": True,
                },
            )
            for event in response["stream"]:
                return json.dumps(event["result"])
        return "Code Interpreter returned no result."

    except Exception as e:
        logger.warning(f"Code Interpreter unavailable, using fallback: {e}")
        tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        tier_discount_pct = tier_rates.get(tier, 0.0)
        tier_discount = round(order_total * tier_discount_pct, 2)
        final_total = round(order_total - tier_discount, 2)
        fallback_result = {
            "points_redeemed": 0,
            "tier_discount_pct": tier_discount_pct,
            "tier_discount": tier_discount,
            "final_total": final_total,
            "remaining_points": loyalty_points,
            "note": "Code Interpreter unavailable — points redemption not calculated",
            "error": str(e),
        }
        return json.dumps(fallback_result)


# ── Agent Entrypoint ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful customer support agent for an Amazon-style
e-commerce store. You can:
- Look up order status and process refunds using the Gateway tools
- Search the knowledge base for product info, return policies, and loyalty details
- Calculate exact loyalty discounts using the loyalty discount tool
- Browse the web when the customer needs live information

Be concise, accurate, and friendly. Use tools rather than guessing at
order-specific or policy-specific details."""


@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    try:
        user_input = payload.get("prompt", "")
        actor_id = payload.get("customer_id", "anonymous")
        session_id = payload.get("session_id") or str(uuid.uuid4())

        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )

        agent_core_browser = AgentCoreBrowser(region=REGION)

        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            agent_core_browser.browser,
        ]

        gateway_client = MCPClient(lambda: streamable_http_client(GATEWAY_URL))
        with gateway_client:
            gateway_tools = gateway_client.list_tools_sync()
            tools.extend(gateway_tools)

            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=SYSTEM_PROMPT,
            )

            response = await agent.invoke_async(user_input)
            return response.message["content"][0]["text"]

    except Exception as e:
        logger.error(f"Agent invocation failed: {e}")
        return f"I'm sorry, something went wrong processing your request: {e}"


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()