import os
import sys
import json
import time
from typing import TypedDict, Annotated, Sequence, Any
import argparse

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

import settings
class C:
    G  = '\\033[92m'
    R  = '\\033[91m'
    Y  = '\\033[93m'
    B  = '\\033[94m'
    CY = '\\033[96m'
    BD = '\\033[1m'
    E  = '\\033[0m'

def init_agents():
    """Load models/indexes and return agent instances."""
    from agents.data_agent import DataAgent
    from agents.policy_rag_agent import PolicyRAGAgent
    from agents.rule_engine_agent import RuleEngineAgent
    from agents.risk_scoring_agent import RiskScoringAgent
    from agents.image_text_agent import ImageTextAgent

    if not os.path.exists(os.path.join(settings.INDEXES_DIR, 'faiss.index')):
        print(f"\\n{C.Y}[!] RAG indexes not found. Building now...{C.E}\\n")
        from rag.build_index import build_indexes
        build_indexes(settings.POLICIES_DIR, settings.INDEXES_DIR, settings.EMBEDDING_MODEL)

    data_agent = DataAgent()
    rag_agent = PolicyRAGAgent(settings.INDEXES_DIR, settings.EMBEDDING_MODEL)
    rule_engine = RuleEngineAgent()
    risk_scorer = RiskScoringAgent()
    image_agent = ImageTextAgent()
    return (data_agent, rag_agent, rule_engine, risk_scorer, image_agent, None)

def print_result(r):
    verdict_color = {'Auto-Approve': C.G, 'Auto-Reject': C.R, 'Escalate': C.Y}.get(r['verdict'], C.E)
    gt_color = C.R if str(r.get('ground_truth', '0')) == '1' else C.G
    
    print(f"\\n{'='*70}\\n {C.BD}RETURN TRIAGE RESULT{C.E}\\n{'='*70}")
    print(f" Return ID  : {C.CY}{r['return_id']}{C.E}")
    print(f" Ground Truth: {r.get('ground_truth', 'unknown')}  -> {verdict_color}{r['verdict'].upper()}{C.E}")
    print(f" ------------------------------------------------------------------")
    print(f" Verdict     : {verdict_color}{r['verdict']}{C.E}")
    print(f" Confidence : {r['confidence']}")
    print(f" Combined   : {r['combined_score']:.3f}  (LightGBM: {r['lgbm_fraud_prob']:.3f}, Rules: {r['rule_score']:.3f})")
    print(f" LLM Called : {'Yes' if r['llm_called'] else 'No'}")
    print(f" Time       : {r['elapsed_seconds']}s")
    print(f" ------------------------------------------------------------------")
    print(f" Reasoning:\\n   {r['reasoning'].replace(chr(10), str(chr(10)+'   '))}")
    print(f"\\n{'='*70}\\n")
class TriageState(TypedDict):
    return_id: str
    image_path: str
    custom_reason: str
    case_data: dict
    risk_results: dict
    rule_results: dict
    image_results: dict
    messages: Annotated[list, "messages"]
    final_verdict: str
    confidence: str
    reasoning: str
    combined_score: float
    num_llm_calls: int

_agents = None

def get_agents():
    # Force reload triggered
    global _agents
    _agents = init_agents()
    return _agents


@tool
def get_relevant_policies(category: str, return_reason: str, is_non_returnable: bool) -> str:
    """Fetches all relevant company policies for a specific return case in a single call. 
    Always call this exactly ONCE before making your decision.
    """
    _, rag_agent, _, _, _, _ = get_agents()
    
    # We automatically construct the optimal hybrid-search query to get everything in one go
    query = f"{category} {return_reason} return {'non-returnable' if is_non_returnable else 'returnable'} policy"
    
    tokenized_query = query.lower().split()
    bm25_scores = rag_agent.bm25.get_scores(tokenized_query)
    bm25_top = bm25_scores.argsort()[::-1][:15]
    
    import numpy as np
    q_emb = rag_agent.encoder.encode([query], normalize_embeddings=True).astype(np.float32)
    _, faiss_indices = rag_agent.faiss_index.search(q_emb, 15)
    faiss_top = faiss_indices[0]
    
    rrf_scores = {}
    rrf_k = 60
    for rank, idx in enumerate(bm25_top):
        rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1.0 / (rrf_k + rank + 1)
    for rank, idx in enumerate(faiss_top):
        if idx >= 0:
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1.0 / (rrf_k + rank + 1)
            
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    results_text = []
    # Increase to top 5 to ensure they get everything they need in one tool call
    for idx, score in sorted_results[:5]:
        chunk = rag_agent.chunks[idx]
        results_text.append(f"[Source: {chunk['source']}]\\n{chunk['text']}")
        
    if not results_text:
        return "No relevant policies found."
    return "\\n\\n".join(results_text)


def fetch_context_node(state: TriageState):
    """Fetches ML and DB context for the return case."""
    data_agent, _, rule_engine, risk_scorer, image_agent, _ = get_agents()
    return_id = state['return_id']
    
    case = data_agent.get_case(return_id)
    rule_results = rule_engine.run(case)
    risk_results = risk_scorer.run(case, top_k=settings.SHAP_TOP_K)
    
    image_results = None
    if state.get("image_path"):
        image_results = image_agent.run(state["image_path"], case, state.get("custom_reason"))
        
    combined = (settings.LGBM_WEIGHT * risk_results['fraud_probability']
                + settings.RULE_WEIGHT * rule_results['rule_score'])
                
    return {
        "case_data": case,
        "rule_results": rule_results,
        "risk_results": risk_results,
        "image_results": image_results,
        "combined_score": combined,
        "messages": []
    }

def build_dossier_message(state: TriageState) -> SystemMessage:
    raw = state['case_data']['raw']
    risk = state['risk_results']
    rules = state['rule_results']
    img = state['image_results']
    reason_display = state.get('custom_reason') or raw.get('reason_category', 'N/A')
    
    dossier = f"""You are a senior return fraud analyst for Flipkart. You must make a decision on this return request.

== RETURN CASE DOSSIER ==

[Case Summary]
- Return ID: {raw.get('return_id', 'N/A')}
- User ID: {raw.get('user_id', 'N/A')}
- Category: {raw.get('category', 'N/A')}
- Product Price: ₹{float(raw.get('price', 0)):,.0f}
- Order Value: ₹{float(raw.get('order_value', 0)):,.0f}
- Discount: {float(raw.get('discount_pct', 0)):.1f}%
- Prepaid Order: {'Yes' if raw.get('is_prepaid') else 'No'}
- Return Reason: {reason_display}
- Return Type: {raw.get('return_type', 'N/A')}
- Image Uploaded: {'Yes' if raw.get('image_uploaded') else 'No'}
- Within Return Window: {'Yes' if raw.get('within_return_window') else 'No'}
- Days Left to Return: {raw.get('days_left_to_return', 'N/A')}
- Days Since Delivered: {raw.get('days_since_delivered', 'N/A')}
- Product Non-Returnable Flag: {'Yes' if raw.get('is_non_returnable') else 'No'}

[User Profile]
- Account Age: {raw.get('account_age_days', 'N/A')} days
- Email Verified: {'Yes' if raw.get('email_verified') else 'No'}
- Shared Device: {'Yes' if raw.get('shared_device_flag') else 'No'}
- Total Orders: {raw.get('total_orders_at_time', 'N/A')}
- Total Prior Returns: {raw.get('total_returns_at_time', 'N/A')}
- Return-to-Order Ratio: {float(raw.get('return_to_order_ratio', 0)):.2f}
- Returns in Last 30 Days: {raw.get('returns_last_30d', 'N/A')}

[Seller Profile]
- Seller Rating: {raw.get('seller_rating', 'N/A')}
- Seller Return Rate: {float(raw.get('seller_return_rate', 0)):.2%}
- Seller Age: {raw.get('seller_age_days', 'N/A')} days

[ML Risk Assessment]
- Fraud Probability (LightGBM): {risk['fraud_probability']:.1%}
- Combined Risk Score: {state['combined_score']:.3f}
- Top contributing features (SHAP):
"""
    for feat in risk['top_shap_features'][:7]:
        shap_str = f"SHAP: {feat['shap_value']:+.3f}" if feat['shap_value'] is not None else ''
        dossier += f"  • {feat['feature']}: {feat['value']:.3f} -> {feat['direction']} ({shap_str})\n"

    dossier += f"\n[Rule Engine Results]\n"
    dossier += f"- Rules Triggered: {rules['rules_triggered_count']}/{rules['total_rules_checked']}\n"
    for r in rules['triggered_rules']:
        dossier += f"  • [{r.rule_id}] {r.rule_name}: {r.description}\n"
    if not rules['triggered_rules']:
        dossier += "  • No rules triggered.\n"

    if img and not img.get('skipped'):
        dossier += "\n[Image Analysis]\n"
        dossier += f"- Consistency Score: {img['consistency_score']}/5\n"
        flags = ', '.join(img.get('red_flags', [])) or 'None'
        dossier += f"- Red Flags: {flags}\n"
        dossier += f"- Assessment: {img.get('assessment', 'N/A')}\n"
        
    dossier += """
== YOUR INSTRUCTIONS ==
1. Use the `get_relevant_policies` tool to fetch ALL relevant policies in a single go. Pass the category, return reason, and whether the item is non-returnable. Do this exactly ONCE.
2. Analyze the evidence strictly against those policies, prioritizing objective evidence over subjective customer claims. 
3. DECISION GUIDELINES:
   - **Auto-Reject Confidently:** If you see strong fraud signals (e.g., Return-to-Order Ratio > 40%, contradictory image evidence, new account claiming non-delivery despite scans, or deceptive image angles), you must choose Auto-Reject. Do NOT blindly trust the customer's stated reason when behavioral or visual evidence strongly contradicts it.
   - **Limit Escalations:** Do not default to Escalate just because a user complains. Escalate ONLY when there is genuine, irreducible ambiguity (e.g., a trusted account with an expired window by 1 day, or a high-value order missing evidence where the policy demands it). Over-escalating hurts our automation rate.
   - **Evidence Precedence:** Weigh inputs in this order: Policy Eligibility -> Confirmed Metadata (scans/images) -> ML Risk/Behavioral History -> Customer Stated Reason.
4. Once you have all the information, output your final decision EXACTLY in this format:
VERDICT: [Auto-Approve / Auto-Reject / Escalate]
CONFIDENCE: [High / Medium / Low]
REASONING:
[Step-by-step chain-of-thought citing policies and features]
"""
    return HumanMessage(content=dossier)

def reasoning_node(state: TriageState):
    """The main LLM node that decides when to use tools and when to output the verdict."""
    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL, 
        temperature=0.1, 
        api_key=settings.GEMINI_API_KEY
    )
    tools = [get_relevant_policies]
    llm_with_tools = llm.bind_tools(tools)
    
    messages = state.get("messages", [])
    if not messages:
        messages = [build_dossier_message(state)]
    
    num_llm_calls = 1
    response = llm_with_tools.invoke(messages)
    
    # Loop to handle multiple sequential tool calls
    while response.tool_calls:
        for tool_call in response.tool_calls:
            print(f" [LLM] Using tool -> get_relevant_policies({tool_call['args']})")
            tool_result = get_relevant_policies.invoke(tool_call["args"])
            
            # Append tool output
            messages.append(response)
            messages.append(ToolMessage(tool_call_id=tool_call["id"], content=tool_result))
            
        # Force LLM to answer again with new context
        response = llm_with_tools.invoke(messages)
        num_llm_calls += 1
        
    text = response.content
        
    if isinstance(text, list):
        text = " ".join([t.get("text", "") for t in text if isinstance(t, dict) and "text" in t])
    elif not isinstance(text, str):
        text = str(text)
        
    import re
    
    verdict = "Escalate"
    verdict_match = re.search(r"VERDICT:\s*(Auto-Approve|Auto-Reject|Escalate)", text, re.IGNORECASE)
    if verdict_match:
        v_str = verdict_match.group(1).lower()
        if "approve" in v_str:
            verdict = "Auto-Approve"
        elif "reject" in v_str:
            verdict = "Auto-Reject"
        
    confidence = "Medium"
    conf_match = re.search(r"CONFIDENCE:\s*(High|Medium|Low)", text, re.IGNORECASE)
    if conf_match:
        confidence = conf_match.group(1).title()
        
    return {
        "final_verdict": verdict,
        "confidence": confidence,
        "reasoning": text,
        "messages": messages,
        "num_llm_calls": num_llm_calls
    }


def build_graph():
    workflow = StateGraph(TriageState)
    
    workflow.add_node("fetch_context", fetch_context_node)
    workflow.add_node("reason", reasoning_node)
    
    workflow.set_entry_point("fetch_context")
    workflow.add_edge("fetch_context", "reason")
    workflow.add_edge("reason", END)
    
    return workflow.compile()


def run_case(return_id: str, image_path: str = None, reason_text: str = None, log_filename: str = 'justification_log.jsonl'):
    app = build_graph()
    
    start = time.time()
    
    state_input = {
        "return_id": return_id,
        "image_path": image_path,
        "custom_reason": reason_text
    }
    
    final_state = app.invoke(state_input)
    elapsed = time.time() - start
    
    
    total_llm_calls = final_state.get('num_llm_calls', 0)
    if final_state.get('image_results') and not final_state['image_results'].get('skipped'):
        total_llm_calls += 1 # Image text agent makes an additional call
        
    result = {
        'return_id': return_id,
        'ground_truth': final_state['case_data']['raw'].get('label', 'unknown'),
        'fraud_type': final_state['case_data']['raw'].get('fraud_type', 'unknown'),
        'verdict': final_state['final_verdict'],
        'confidence': final_state['confidence'],
        'reasoning': final_state['reasoning'],
        'llm_called': True,
        'num_llm_calls': total_llm_calls,
        'combined_score': final_state['combined_score'],
        'lgbm_fraud_prob': final_state['risk_results']['fraud_probability'],
        'rule_score': final_state['rule_results']['rule_score'],
        'rules_triggered': [r.to_dict() for r in final_state['rule_results']['triggered_rules']],
        'top_shap': final_state['risk_results']['top_shap_features'][:5],
        # Empty since we let the LLM search it via MCP tools dynamically
        'rag_policies': [], 
        'elapsed_seconds': round(elapsed, 2),
    }
    
    print_result(result)
    
    # The problem statement requires a comprehensive output demonstrating the reasoning chain.
    log_path = os.path.join(settings.OUTPUT_DIR, log_filename)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    
    # Include all input evidence for complete auditability
    log_entry = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'return_id': return_id,
        'inputs': {
            'case_data': final_state['case_data']['raw'],
            'risk_score': final_state['risk_results']['fraud_probability'],
            'shap_features': final_state['risk_results']['top_shap_features'],
            'rule_score': final_state['rule_results']['rule_score'],
            'triggered_rules': [r.to_dict() for r in final_state['rule_results']['triggered_rules']]
        },
        'outputs': {
            'verdict': final_state['final_verdict'],
            'confidence': final_state['confidence'],
            'reasoning_chain': final_state['reasoning']
        }
    }
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, default=str) + '\n')
        
    print(f"\\n[Logger] Case {return_id} decision and full evidence chain saved to {log_path}")

    # Save escalation
    if final_state['final_verdict'] == 'Escalate':
        escalation_path = os.path.join(settings.OUTPUT_DIR, 'escalation_queue.jsonl')
        with open(escalation_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, default=str) + '\n')

    
    return result
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--return-id', type=str, help='Return ID to process (e.g. RET_0001234)')
    args = parser.parse_args()
    
    if not settings.GEMINI_API_KEY:
        print("GEMINI_API_KEY not set!")
        sys.exit(1)
        
    print("Initializing Agents for LangGraph Orchestration...")
    get_agents()
    
    if args.return_id:
        run_case(args.return_id)
    else:
        # Default to a random test case
        print("No --return-id provided. Running a random test case...")
        data_agent = get_agents()[0]
        test_ids = data_agent.get_test_return_ids()
        import random
        rid = random.choice(test_ids)
        run_case(rid)
