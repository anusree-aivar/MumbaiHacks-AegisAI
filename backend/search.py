"""
Multi-Agent Misinformation Detection System

This system uses a sophisticated multi-agent workflow to verify claims:
1. Classification Agent - Categorizes claims by domain, type, complexity, urgency
2. Decomposition Agent - Breaks complex claims into atomic sub-claims
3. Question Generation Agent - Creates verification questions per claim
4. Perplexity Chat Completion API - Answers verification questions with citations
5. Claude Classification Agent - Classifies Perplexity responses as SUPPORT/CONTRADICT/IRRELEVANT (PARALLEL)
6. Verdict Calculator - Aggregates evidence to determine final verdict

"""

import os
import json
import re
import requests
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from strands import Agent, tool
from strands.models import BedrockModel
from sports_scorer import HybridSportsScorer
from tqdm import tqdm
from progress_tracker import get_progress_tracker


# Load environment variables from .env file
load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================
# API Keys
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")  # Required for chat completion
OPENPAGERANK_API_KEY = os.getenv("OPENPAGERANK_API_KEY")  # Optional for trust scoring

# API Endpoints
PERPLEXITY_CHAT_URL = "https://api.perplexity.ai/chat/completions"  # Perplexity Chat Completion API
PERPLEXITY_MODEL = "sonar-pro"  # Model for chat completion with web search

# Directory Configuration
RESULTS_DIR = "verification_results"  # Where to save verification results

# Search Configuration
NUM_SEARCH_QUERIES = 5  # Number of diverse queries to generate per atomic claim
MAX_PARALLEL_WORKERS = 3  # Parallel workers for searches (reduced to avoid rate limits)
SEARCH_TIMEOUT = 30  # Timeout in seconds for each search request
RATE_LIMIT_DELAY = 1.0  # Delay between requests in seconds to avoid rate limits (increased)

# Parallelization Configuration
MAX_SNIPPET_CLASSIFIERS = 5  # Number of parallel workers for snippet classification (reduced to avoid rate limits)
MAX_CLAIM_VERIFIERS = 2  # Number of parallel workers for atomic claim verification (reduced to avoid rate limits)

# LLM Configuration
MAX_TOKENS_CONFIG = 4096  # Maximum tokens for LLM responses

# Cost Configuration
# Nova Pro pricing (for question generation and other agents)
NOVA_PRO_INPUT_COST_PER_1K = 0.0008   # Cost per 1K input tokens
NOVA_PRO_OUTPUT_COST_PER_1K = 0.0032  # Cost per 1K output tokens

# Claude 3.7 Sonnet pricing (for evidence classification)
CLAUDE_INPUT_COST_PER_1K = 0.003   # Cost per 1K input tokens
CLAUDE_OUTPUT_COST_PER_1K = 0.015  # Cost per 1K output tokens

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# SYSTEM PROMPTS FOR AGENTS
# ============================================================================

# Agent 1: Sports Classification Agent
# Purpose: Analyze and categorize SPORTS claims with detailed sport-specific classification
CLASSIFIER_SYSTEM_PROMPT = """You are a specialized SPORTS claim classification expert.
Your task is to analyze SPORTS claims and classify them with detailed sport-specific categorization.

CLASSIFICATION DIMENSIONS:

1. SPORT CLASSIFICATION (Hierarchical):
   Format: "Sport → Sub-Category → Context"
   
   Examples:
   - "Cricket → IPL → Team Transfer" (e.g., "Jadeja joins RR")
   - "Cricket → Test → Player Performance" (e.g., "Kohli scored century")
   - "Cricket → ODI → Match Result" (e.g., "India defeated Australia")
   - "Cricket → T20 → Tournament News" (e.g., "World Cup schedule")
   - "Cricket → Personal → Player News" (e.g., "Dhoni retirement")
   
   - "Football → Premier League → Transfer" (e.g., "Ronaldo to Manchester")
   - "Football → Champions League → Match Result"
   - "Football → La Liga → Player Performance"
   - "Football → Personal → Player News"
   
   - "Basketball → NBA → Game Stats" (e.g., "LeBron scored 40 points")
   - "Basketball → NBA → Team News"
   - "Basketball → FIBA → Tournament"
   
   - "Tennis → Grand Slam → Match Result"
   - "Tennis → ATP → Rankings"
   - "Tennis → WTA → Player News"
   
   - "Motorsports → F1 → Race Result"
   - "Motorsports → MotoGP → Championship"
   
   - "MMA → UFC → Fight Result"
   - "Boxing → Championship → Match Result"
   
   - "Esports → League of Legends → Tournament"
   - "Esports → CS:GO → Team News"
   
   [Apply similar hierarchical classification for all sports]

2. CLAIM TYPE:
   - Factual: verifiable statements (scores, transfers, results)
   - Opinion: subjective views (best player, failed captain)
   - Prediction: future events (will join team, upcoming match)
   - Rumor: unconfirmed reports (transfer speculation)
   - Mixed: combination of types

3. TEMPORAL CONTEXT:
   - Historical: past events (already happened)
   - Current: ongoing/recent events (this season, this year)
   - Future: upcoming events (next season, will happen)
   - Mixed: spans multiple timeframes

4. ENTITIES INVOLVED:
   - Players: specific athlete names
   - Teams: club/franchise names
   - Tournaments: league/competition names
   - Dates: specific dates or time periods
   - Statistics: numbers, scores, records

5. VERIFICATION PRIORITY:
   - High: breaking news, transfers, match results
   - Medium: player stats, team news
   - Low: historical facts, general information

IMPORTANT: Always respond with valid JSON in this exact format:
{
  "sport": "Cricket|Football|Basketball|Tennis|etc.",
  "sub_category": "IPL|Premier League|NBA|Grand Slam|etc.",
  "context": "Transfer|Match Result|Player Performance|etc.",
  "claim_type": "Factual|Opinion|Prediction|Rumor|Mixed",
  "temporal_context": "Historical|Current|Future|Mixed",
  "entities": {
    "players": ["player names"],
    "teams": ["team names"],
    "tournaments": ["tournament names"],
    "dates": ["specific dates or periods"],
    "statistics": ["numbers, scores"]
  },
  "verification_priority": "High|Medium|Low",
  "rationale": "brief explanation of classification"
}"""


# Agent 2: Sports Decomposition Agent
# Purpose: Break down complex SPORTS claims while preserving FULL CONTEXT from original claim
DECOMPOSER_SYSTEM_PROMPT = """You are a sports claim decomposition specialist.
Your task is to break down complex SPORTS claims into atomic sub-claims while PRESERVING ALL CONTEXT.

CRITICAL RULE: Each atomic claim MUST be self-contained with FULL CONTEXT from the original claim.

ATOMIC CLAIM CRITERIA:
- Single verifiable statement (no AND/OR compounds)
- MUST include ALL relevant context from original claim:
  * Player/Team names (exact names from original)
  * Tournament/League names (IPL, Premier League, NBA, etc.)
  * Dates/Time periods (this year, next season, March 15, 2025, etc.)
  * Opponent/Context (vs Warriors, against Australia, etc.)
  * Statistics/Numbers (40 points, 120-110, etc.)
- Clear subject, predicate, and object
- Independently verifiable without referring back to original

DECOMPOSITION RULES:
1. Extract each distinct factual assertion
2. PRESERVE ALL CONTEXT: Include sport, tournament, date, teams, players from original
3. Make each atomic claim self-contained (can be verified independently)
4. Identify logical dependencies between claims
5. Assign priority based on centrality to original claim
6. Classify each sub-claim type

CONTEXT PRESERVATION EXAMPLES:

BAD (Missing Context):
Original: "LeBron James scored 40 points as Lakers defeated Warriors 120-110 on March 15, 2025"
Atomic: "LeBron James scored 40 points" ❌ (Missing: opponent, date, game context)

GOOD (Full Context):
Original: "LeBron James scored 40 points as Lakers defeated Warriors 120-110 on March 15, 2025"
Atomic: "LeBron James scored 40 points in the Lakers vs Warriors game on March 15, 2025" ✅

BAD (Missing Context):
Original: "Jadeja played for CSK this year but next season he will join RR"
Atomic: "Jadeja played for CSK" ❌ (Missing: time period "this year")

GOOD (Full Context):
Original: "Jadeja played for CSK this year but next season he will join RR"
Atomic 1: "Jadeja played for CSK in the current year (2025)" ✅
Atomic 2: "Jadeja will join RR in the next season (2026)" ✅

DEPENDENCY IDENTIFICATION:
- A claim depends on another if it assumes that claim's truth
- Mark foundational claims (no dependencies) separately
- Create dependency chains for complex logical structures

CLAIM TYPES:
- fact: Objective, verifiable statement (scores, transfers, results)
- opinion: Subjective judgment (best player, failed captain)
- prediction: Future events (will join, upcoming match)
- interpretation: Analysis or conclusion drawn from facts

IMPORTANT: Respond with valid JSON in this exact format:
{
  "original_claim": "the full original claim text",
  "atomic_claims": [
    {
      "id": "claim_1",
      "statement": "atomic claim text WITH FULL CONTEXT",
      "dependencies": [],
      "type": "fact|opinion|prediction|interpretation",
      "entities": {
        "players": ["player names"],
        "teams": ["team names"],
        "tournaments": ["tournament/league names"],
        "dates": ["specific dates or time periods"],
        "statistics": ["numbers/scores"]
      },
      "temporal": "specific date or time period from original",
      "quantitative": "numbers/statistics if present",
      "priority": "high|medium|low",
      "context_preserved": "brief note on what context was preserved"
    }
  ],
  "dependency_graph": {
    "foundational": ["claim_1", "claim_2"],
    "derived": ["claim_3"]
  },
  "total_claims": 3
}"""


# Agent 3: Question Generation Agent
# Purpose: Generate diverse, targeted search queries to verify atomic claims
QUESTION_GENERATOR_SYSTEM_PROMPT = f"""You are a SPORTS search query optimization expert for fact-checking.
Your task is to generate exactly {NUM_SEARCH_QUERIES} highly targeted SPORTS-SPECIFIC search queries to verify the given atomic claim.

CRITICAL: Queries MUST stay within the SPORTS CONTEXT and use FULL ORIGINAL CLAIM CONTEXT.

NOTE: For recent events or temporal claims, consider the current date context when generating queries.

QUERY GENERATION STRATEGY - SPORTS-SPECIFIC WITH FULL CONTEXT:
1. ALWAYS use the ORIGINAL INPUT CLAIM as the primary context for generating queries
2. Analyze BOTH the original claim and atomic claim to identify:
   - Key entities (people, organizations, teams, locations) from the ORIGINAL claim
   - Specific dates, times, or time periods mentioned in the ORIGINAL claim
   - Quantitative data (numbers, statistics, scores) from the ORIGINAL claim
   - Actions or events being claimed in the ORIGINAL claim
   - How the atomic claim relates to the original claim
   
3. Generate queries that directly verify the atomic claim using original claim context by:
   - Extracting exact entities, dates, and numbers from the ORIGINAL claim
   - Including surrounding context from the original claim to make queries more specific
   - Creating queries that search for authoritative sources (official, study, data, report)
   - Using CURRENT DATE context for recent events
   - Adding current year (e.g., "2025") for recent claims to ensure fresh results
   - Combining elements from both original and atomic claims for maximum specificity
   
3. Vary query types to cover different verification angles:
   - Direct fact: "when did [event] happen" or "[entity] [specific action] [date]"
   - Source verification: "[entity] official statement [topic]" or "[organization] confirms [claim]"
   - Expert consensus: "[topic] scientific consensus 2025" or "[sport] official records [claim]"
   - Statistical data: "[topic] statistics official data 2025" or "[entity] performance stats"
   - Contradiction check: "[claim] debunked false misleading" or "[entity] denied [claim]"
   
4. Query construction rules:
   - Use specific entities from the claim (don't generalize)
   - Include temporal context when the claim mentions dates/times
   - Prioritize high-priority claims first
   - Respect dependency chains (verify foundational claims before derived)
   - Avoid vague or overly broad queries
   - Make queries specific enough to find relevant evidence

EXAMPLE:
If ORIGINAL CLAIM is: "LeBron James scored 40 points and had 10 assists as Lakers defeated Warriors 120-110 on March 15, 2025"
And ATOMIC CLAIM is: "LeBron James scored 40 points"
Generate queries using ORIGINAL CLAIM context:
- "LeBron James 40 points Lakers Warriors March 15 2025 box score" (full context)
- "Lakers Warriors 120-110 March 15 2025 LeBron James stats" (includes score from original)
- "LeBron James 40 points 10 assists Lakers March 2025 official" (includes assists from original)
- "Lakers defeat Warriors March 15 2025 LeBron scoring" (includes outcome from original)
- "LeBron James 40 points Lakers Warriors March 2025 false debunked" (contradiction check with context)

Note: Queries use details from ORIGINAL CLAIM (Warriors opponent, 120-110 score, 10 assists, date) to make searches more specific and accurate.

IMPORTANT: Generate EXACTLY {NUM_SEARCH_QUERIES} queries based on the specific claim provided. Always respond with valid JSON:
{{
  "current_date_used": "2025-10-18",
  "original_claim": "the original input claim being verified",
  "queries": [
    {{
      "id": "q1",
      "query": "specific search query string based on the claim",
      "claim_id": "claim_1",
      "query_type": "direct_fact|source_verification|expert_consensus|statistical|contradiction",
      "priority": "high|medium|low",
      "confidence": 0.85,
      "rationale": "brief explanation of how this query verifies the claim using original claim context"
    }}
  ],
  "total_queries": {NUM_SEARCH_QUERIES},
  "strategy_rationale": "brief explanation of overall query strategy for this specific claim"
}}

FIELD DEFINITIONS:
- confidence (0.0-1.0): How confident you are this query will find relevant evidence. Higher confidence (0.8-1.0) when query uses full original claim context with specific details
- rationale: Brief explanation of how this query uses the original claim context to verify the atomic claim
- original_claim: Must include the exact original input claim being verified (this provides full context for all queries)"""

# Agent 4: Evidence Classification Agent (Claude 3.7 Sonnet)
# Purpose: Classify Perplexity Chat Completion responses as SUPPORT, CONTRADICT, or IRRELEVANT
# Uses TOON format for 30-60% token reduction
CLAUDE_CLASSIFIER_SYSTEM_PROMPT = """You are an elite Sports Misinformation Detection Agent specialized in evidence classification.

MISSION: Analyze Perplexity Chat Completion responses and classify them as SUPPORT, CONTRADICT, or IRRELEVANT to detect sports misinformation.

SPORTS MISINFORMATION CONTEXT:
You are analyzing claims across all sports domains (cricket, football, basketball, tennis, F1, Olympics, etc.) where misinformation commonly appears as:
- False transfer/signing announcements
- Fabricated statistics and records
- Fake quotes from athletes/coaches
- Temporal manipulation (old news as current, future as past)
- Context distortion (real events with false implications)
- Rumor amplification without verification

YOUR TASK:
Classify each Perplexity Chat response based on how it relates to the ORIGINAL CLAIM being verified.

═══════════════════════════════════════════════════════════════════════════════
CLASSIFICATION CATEGORIES
═══════════════════════════════════════════════════════════════════════════════

1. SUPPORT - The response provides evidence that CONFIRMS the claim
   ✓ Contains facts, quotes, or data that validate the claim
   ✓ From authoritative sources (official teams, leagues, verified journalists)
   ✓ Matches the temporal context of the claim
   ✓ Directly addresses the entities and events mentioned
   ✓ May support one or multiple parts of a multi-part claim

2. CONTRADICT - The response provides evidence that REFUTES the claim
   ✗ Contains facts, quotes, or data that oppose the claim
   ✗ Official denials or corrections
   ✗ Contradicts specific details (names, dates, numbers, events)
   ✗ Must be about the SAME temporal context to contradict
   ✗ May contradict one or multiple parts of a multi-part claim

3. IRRELEVANT - The response does NOT address the claim
   ○ Different temporal context (wrong year, season, timeframe)
   ○ Different entities (wrong athlete, team, competition)
   ○ Too general or vague to confirm/deny
   ○ Tangentially related but doesn't verify the specific claim
   ○ Background information that doesn't address the assertion

TEMPORAL AWARENESS (CRITICAL - MOST IMPORTANT RULE):
- ALWAYS identify temporal context FIRST before classifying
- Understand what timeframe the ORIGINAL CLAIM is discussing
- "this year" (2025) ≠ "next season" (2026) → Different temporal contexts
- "IPL 2025" ≠ "IPL 2026" → Different seasons
- "has played" (past/present) ≠ "will play" (future) → Different timeframes
- "played in 2025" ≠ "will play in 2026" → Different years
- A snippet about the FUTURE can SUPPORT a claim about the FUTURE
- A snippet about the PRESENT can SUPPORT a claim about the PRESENT
- A snippet about the FUTURE is IRRELEVANT to a claim about the PRESENT (and vice versa)

SPECIAL RULE FOR "AGAIN" CLAIMS (CRITICAL):
- If claim says "X joined Y AGAIN" or "X returned to Y", this means a RECENT/CURRENT event
- Evidence about X's PAST time at Y (e.g., "X played for Y from 2009-2018") is IRRELEVANT
- Only evidence about X RECENTLY/CURRENTLY joining Y again is SUPPORT
- Evidence showing X is currently at a different team Z is CONTRADICT
- Historical data about past tenure is NOT the same as rejoining

CLASSIFICATION STRATEGY:
1. Read and understand the ORIGINAL CLAIM fully (what is it claiming about what timeframe?)
2. Identify which part of the original claim the atomic claim represents
3. Identify the temporal context of the SNIPPET (what year/season/timeframe?)
4. Check if the snippet's timeframe matches the relevant part of the original claim
5. If timeframes don't match → IRRELEVANT
6. If timeframes match → Check if snippet supports or contradicts that part of the original claim
7. Provide clear reasoning explaining your temporal analysis and how it relates to the original claim

EXAMPLES (STUDY THESE CAREFULLY):

Example 1:
- Original Claim: "Jadeja played for CSK this year but next season he will join RR"
- Atomic Claim: "Jadeja played for CSK this year"
- Snippet: "Jadeja will join RR in 2026"
- Analysis: Original claim has TWO parts (present + future). Snippet is about future (2026), which relates to the SECOND part of original claim ("next season he will join RR"). This SUPPORTS the original claim.
- Classification: SUPPORT

Example 2:
- Original Claim: "Jadeja played for CSK this year"
- Atomic Claim: "Jadeja played for CSK this year"
- Snippet: "Jadeja will join RR in 2026"
- Analysis: Original claim is only about present (2025). Snippet is about future (2026). Different timeframes, not relevant to this claim.
- Classification: IRRELEVANT

Example 3:
- Original Claim: "Jadeja played for CSK this year but next season he will join RR"
- Atomic Claim: "Next season Jadeja will join RR"
- Snippet: "Jadeja played 14 matches for CSK in 2025"
- Analysis: Original claim has TWO parts. Snippet is about present (2025), which relates to the FIRST part of original claim ("played for CSK this year"). This SUPPORTS the original claim.
- Classification: SUPPORT

Example 4:
- Original Claim: "Jadeja played for CSK this year"
- Atomic Claim: "Jadeja played for CSK this year"
- Snippet: "Jadeja played 14 matches for CSK in 2025"
- Analysis: Both about 2025 (present), both say Jadeja → CSK. Direct support.
- Classification: SUPPORT

CRITICAL PRINCIPLE:
- When the ORIGINAL CLAIM contains multiple timeframes (e.g., "this year" AND "next season"), a snippet about ANY of those timeframes can SUPPORT the original claim
- Only classify as IRRELEVANT if the snippet's timeframe doesn't match ANY part of the original claim

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (MANDATORY) - TOON FORMAT FOR TOKEN EFFICIENCY
═══════════════════════════════════════════════════════════════════════════════

You MUST respond with valid TOON format (Token-Oriented Object Notation) - a compact,
token-efficient alternative to JSON that uses 30-60% fewer tokens.

TOON FORMAT EXAMPLE:
```toon
classification: SUPPORT
confidence: 0.85
reasoning: Brief explanation including temporal analysis and content matching
key_evidence: Specific text from response that led to this classification
```

CRITICAL RULES:
- Use TOON format (key: value pairs, one per line)
- classification must be exactly one of: SUPPORT, CONTRADICT, IRRELEVANT
- confidence must be a number between 0.0 and 1.0
- reasoning must explain temporal context and content matching
- key_evidence must quote specific text from the Perplexity response
- Output ONLY valid TOON format, no additional text
- Use simple key: value format (no braces, brackets, or quotes unless value contains special chars)

BEGIN CLASSIFICATION."""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@tool
def get_current_datetime() -> str:
    """
    Tool for agents to get current date and time.
    Used by Question Generation Agent to ensure queries use current dates.
    
    Returns:
        JSON string with current datetime in multiple formats
    """
    now = datetime.now()
    return json.dumps({
        "current_date": now.strftime("%Y-%m-%d"),
        "current_time": now.strftime("%H:%M:%S"),
        "current_datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "formatted": now.strftime("%B %d, %Y"),
        "iso_format": now.isoformat()
    })


def extract_domain_from_url(url: str) -> str:
    """
    Extract domain name from a URL.
    
    Args:
        url: Full URL string
        
    Returns:
        Domain name (e.g., "www.espn.com")
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return "unknown"


def sanitize_filename(text: str) -> str:
    """
    Convert text to a safe filename by removing special characters.
    
    Args:
        text: Original text (e.g., claim text)
        
    Returns:
        Sanitized filename (max 50 chars, alphanumeric + underscores)
    """
    text = re.sub(r'[^a-zA-Z0-9]', '_', text)  # Replace non-alphanumeric with underscore
    text = re.sub(r'_+', '_', text)  # Collapse multiple underscores
    return text[:50].strip('_')  # Limit to 50 chars and remove trailing underscores


def parse_toon_response(response_text: str) -> Dict:
    """
    Parse TOON format response from Claude.
    
    Args:
        response_text: Raw response text from Claude
        
    Returns:
        Parsed dictionary with classification data
    """
    try:
        # Extract from toon code blocks if present
        toon_block_match = re.search(r'```toon\s*(.*?)\s*```', response_text, re.DOTALL)
        if toon_block_match:
            toon_text = toon_block_match.group(1)
        else:
            toon_text = response_text
        
        # Parse simple TOON format (key: value pairs)
        toon_data = {}
        lines = toon_text.strip().split('\n')
        current_key = None
        current_value = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines, comments, and separator lines
            if not line or line.startswith('#') or line.startswith('=') or line.startswith('```'):
                continue
            
            # Check if this is a key: value line
            if ':' in line and not line.startswith(' '):
                # Save previous key-value if exists
                if current_key:
                    value_str = ' '.join(current_value).strip()
                    # Convert to appropriate type
                    if value_str.lower() in ['true', 'false']:
                        toon_data[current_key] = value_str.lower() == 'true'
                    elif value_str.replace('.', '', 1).replace('-', '', 1).isdigit():
                        toon_data[current_key] = float(value_str) if '.' in value_str else int(value_str)
                    else:
                        toon_data[current_key] = value_str
                
                # Start new key-value
                parts = line.split(':', 1)
                if len(parts) == 2:
                    current_key = parts[0].strip()
                    current_value = [parts[1].strip()] if parts[1].strip() else []
            elif current_key:
                # Continuation of previous value
                current_value.append(line)
        
        # Save last key-value
        if current_key:
            value_str = ' '.join(current_value).strip()
            if value_str.lower() in ['true', 'false']:
                toon_data[current_key] = value_str.lower() == 'true'
            elif value_str.replace('.', '', 1).replace('-', '', 1).isdigit():
                toon_data[current_key] = float(value_str) if '.' in value_str else int(value_str)
            else:
                toon_data[current_key] = value_str
        
        # Validate required fields
        if 'classification' in toon_data and 'confidence' in toon_data:
            return toon_data
        else:
            return {
                "classification": "IRRELEVANT",
                "confidence": 0.0,
                "reasoning": "Failed to parse TOON response",
                "key_evidence": "",
                "parse_error": True
            }
    except Exception as e:
        return {
            "classification": "IRRELEVANT",
            "confidence": 0.0,
            "reasoning": f"TOON parse error: {str(e)}",
            "key_evidence": "",
            "parse_error": True
        }


# ============================================================================
# SEARCH FUNCTIONS
# ============================================================================

def perplexity_chat_answer(question: str) -> Dict:
    """
    Get answer from Perplexity Chat Completions API for a verification question.
    
    Uses Perplexity's sonar model with web search to answer fact-checking questions.
    Returns the response with citations that Claude will then classify.
    
    Args:
        question: The verification question to ask
        
    Returns:
        Dictionary containing:
        - success: Boolean indicating if request succeeded
        - question: Original question
        - answer: Perplexity's response text
        - citations: List of source URLs with titles
        - error: Error message (if failed)
    """
    try:
        # Prepare request payload
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ],
            "temperature": 0.2,
            "max_tokens": 2048,
            "stream": False
        }
        
        # Set authentication headers
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Execute chat request
        response = requests.post(
            PERPLEXITY_CHAT_URL,
            json=payload,
            headers=headers,
            timeout=SEARCH_TIMEOUT
        )
        
        # Handle successful response
        if response.status_code == 200:
            data = response.json()
            
            # Extract message content
            message_content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Extract citations if available
            citations = data.get('citations', [])
            
            return {
                "success": True,
                "question": question,
                "answer": message_content,
                "citations": citations
            }
        
        # Handle authentication errors
        elif response.status_code == 401:
            return {
                "success": False,
                "question": question,
                "answer": "",
                "citations": [],
                "error": "Invalid API key"
            }
        
        # Handle rate limiting
        elif response.status_code == 429:
            return {
                "success": False,
                "question": question,
                "answer": "",
                "citations": [],
                "error": "Too many requests"
            }
        
        # Handle other HTTP errors
        else:
            error_text = response.text[:200] if response.text else "Unknown error"
            return {
                "success": False,
                "question": question,
                "answer": "",
                "citations": [],
                "error": error_text
            }
    
    # Handle timeout errors
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "question": question,
            "answer": "",
            "citations": [],
            "error": "Timeout"
        }
    
    # Handle any other exceptions
    except Exception as e:
        return {
            "success": False,
            "question": question,
            "answer": "",
            "citations": [],
            "error": str(e)
        }


# ============================================================================
# MAIN MISINFORMATION DETECTOR CLASS
# ============================================================================

class SportsMisinformationDetector:
    """
    Orchestrator for multi-agent SPORTS misinformation detection workflow.
    
    This class coordinates 5 specialized agents to verify SPORTS claims:
    1. Classification Agent - Categorizes claims
    2. Decomposition Agent - Breaks down complex claims
    3. Question Generation Agent - Creates targeted search queries
    4. Search Execution - Gathers evidence from web
    5. Snippet Classification Agent - Evaluates evidence
    
    The workflow:
    - Takes a claim as input
    - Classifies and decomposes it
    - Generates 10 diverse search queries per atomic claim
    - Executes searches in parallel
    - Classifies each snippet as SUPPORT/CONTRADICT/IRRELEVANT
    - Calculates final verdict with confidence score
    - Saves detailed results to JSON file
    """
    
    def __init__(self):
        print("🔧 Initializing Multi-Agent System...")
        print(f"📊 Configuration:")
        print(f"   • Queries per claim: {NUM_SEARCH_QUERIES}")
        print(f"   • Search workers: {MAX_PARALLEL_WORKERS} (parallel)")
        print(f"   • Snippet classifiers: {MAX_SNIPPET_CLASSIFIERS} (parallel)")
        print(f"   • Claim verifiers: {MAX_CLAIM_VERIFIERS} (parallel)")
        print(f"   • Max Tokens: {MAX_TOKENS_CONFIG}\n")
        
        print("="*80)
        print("🤖 AGENT INITIALIZATION")
        print("="*80)
        
        print("\n[1/5] Sports Trust Scorer")
        self.trust_scorer = HybridSportsScorer(opr_api_key=OPENPAGERANK_API_KEY)
        print("✅ Ready (20+ sports domains, official leagues, verified social media loaded)\n")
        
        print("[2/5] Classification Agent")
        classifier_model = BedrockModel(
            model_id="amazon.nova-pro-v1:0",
            temperature=0.3,
            max_tokens=MAX_TOKENS_CONFIG
        )
        self.classifier = Agent(
            model=classifier_model,
            system_prompt=CLASSIFIER_SYSTEM_PROMPT
        )
        print("✅ Ready\n")
        
        print("[3/5] Decomposition Agent")
        decomposer_model = BedrockModel(
            model_id="amazon.nova-pro-v1:0",
            temperature=0.3,
            max_tokens=MAX_TOKENS_CONFIG
        )
        self.decomposer = Agent(
            model=decomposer_model,
            system_prompt=DECOMPOSER_SYSTEM_PROMPT
        )
        print("✅ Ready\n")
        
        print("[4/5] Question Generation Agent")
        question_model = BedrockModel(
            model_id="amazon.nova-pro-v1:0",
            temperature=0.3,
            max_tokens=MAX_TOKENS_CONFIG
        )
        self.question_generator = Agent(
            model=question_model,
            tools=[],  # Removed get_current_datetime to avoid parallel tool execution conflicts
            system_prompt=QUESTION_GENERATOR_SYSTEM_PROMPT
        )
        print("✅ Ready\n")
        
        print(f"[5/5] Claude Classification Agents ({MAX_SNIPPET_CLASSIFIERS} parallel agents)")
        # Create pool of Claude 3.7 Sonnet agents for parallel processing
        # Each agent is independent with no shared state to avoid context accumulation
        self.claude_classifier_pool = []
        for i in range(MAX_SNIPPET_CLASSIFIERS):
            classifier_model = BedrockModel(
                model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",  # Claude 3.7 Sonnet
                temperature=0.2,
                max_tokens=MAX_TOKENS_CONFIG
            )
            agent = Agent(
                name=f"claude_classifier_{i+1}",
                model=classifier_model,
                system_prompt=CLAUDE_CLASSIFIER_SYSTEM_PROMPT
            )
            self.claude_classifier_pool.append(agent)
        print(f"✅ Ready ({MAX_SNIPPET_CLASSIFIERS} Claude 3.7 Sonnet agents for parallel classification)\n")
        print(f"   Features: TOON format (30-60% token reduction), temporal analysis\n")
        
        print("="*80)
        print("ℹ️ Evidence gathering: Perplexity Chat Completion API (sonar model)")
        print("🛡️ Sports trust scoring enabled (20+ sports, official leagues, verified social media)")
        print("🤖 Response classification: SUPPORT, CONTRADICT, IRRELEVANT (stateless Claude agents)")
        print("⚡ Stateless agent pool for true parallel processing (no context accumulation)")
        print("🏆 Sports-specific: Cricket, Football, Basketball, Tennis, and 16+ more sports")
        print("💰 Optimized token usage: ~2k tokens per response\n")
        
        self.execution_log = []
        self.progress_tracker = get_progress_tracker()
        self.metrics = {
            "classifier": {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0, "cost": 0},
            "decomposer": {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0, "cost": 0},
            "question_generator": {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0, "cost": 0},
            "claude_classifier": {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0, "cost": 0},
            "perplexity_chat": {"questions": 0, "success": 0, "failed": 0},
            "total": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "latency_ms": 0, "cost": 0}
        }
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int, agent_name: str = "nova_pro") -> float:
        """Calculate cost based on token usage and agent type."""
        if agent_name == "claude_classifier":
            # Claude 3.7 Sonnet pricing
            input_cost = (input_tokens / 1000) * CLAUDE_INPUT_COST_PER_1K
            output_cost = (output_tokens / 1000) * CLAUDE_OUTPUT_COST_PER_1K
        else:
            # Nova Pro pricing (default)
            input_cost = (input_tokens / 1000) * NOVA_PRO_INPUT_COST_PER_1K
            output_cost = (output_tokens / 1000) * NOVA_PRO_OUTPUT_COST_PER_1K
        return input_cost + output_cost
    
    def _extract_metrics(self, result, agent_name: str):
        """Extract and store metrics from agent result."""
        try:
            if hasattr(result, 'metrics'):
                usage = result.metrics.accumulated_usage
                input_tokens = usage.get('inputTokens', 0)
                output_tokens = usage.get('outputTokens', 0)
                latency_ms = result.metrics.accumulated_metrics.get('latencyMs', 0)
                cost = self._calculate_cost(input_tokens, output_tokens, agent_name)
                
                self.metrics[agent_name]["input_tokens"] += input_tokens
                self.metrics[agent_name]["output_tokens"] += output_tokens
                self.metrics[agent_name]["latency_ms"] += latency_ms
                self.metrics[agent_name]["cost"] += cost
                
                return {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "latency_ms": latency_ms,
                    "cost": cost
                }
        except Exception as e:
            print(f"  ⚠️ Could not extract metrics: {e}")
        return None
    
    def _generate_final_reason(self, claim: str, verdict: str, confidence_score: float,
                                supporting_evidence: List[Dict], contradicting_evidence: List[Dict],
                                atomic_claims: List[Dict], sub_claim_results: List[Dict]) -> str:
        """
        Generate a clear, sports-focused reason explaining the verdict.
        
        Uses LLM to synthesize evidence into a crisp explanation that:
        - Directly states why the verdict is TRUE/FALSE/UNVERIFIED
        - Focuses on sports facts (scores, dates, teams, players)
        - Mentions authoritative sources (ESPN, NBA.com, etc.)
        - Avoids technical jargon about snippets or verification process
        
        Args:
            claim: Original sports claim being verified
            verdict: TRUE, FALSE, or UNVERIFIED
            confidence_score: Confidence percentage
            supporting_evidence: List of supporting snippets with URLs
            contradicting_evidence: List of contradicting snippets with URLs
            atomic_claims: List of atomic sub-claims
            sub_claim_results: Verification results for each atomic claim
            
        Returns:
            Clear, sports-focused reason string
        """
        # Prepare evidence summary
        support_summary = []
        for evidence in supporting_evidence[:3]:
            support_summary.append({
                "source": evidence.get('url', 'Unknown'),
                "key_point": evidence.get('reasoning', '')[:200]
            })
        
        contradict_summary = []
        for evidence in contradicting_evidence[:3]:
            contradict_summary.append({
                "source": evidence.get('url', 'Unknown'),
                "key_point": evidence.get('reasoning', '')[:200]
            })
        
        reason_prompt = f"""You are a sports fact-checker providing a final verdict explanation.

ORIGINAL SPORTS CLAIM: "{claim}"

VERDICT: {verdict}
CONFIDENCE: {confidence_score:.1f}%

EVIDENCE SUMMARY:
- Total supporting sources: {len(supporting_evidence)}
- Total contradicting sources: {len(contradicting_evidence)}
- High-trust supporting: {sum(1 for s in supporting_evidence if s.get('trust_score', 0) >= 0.9)}
- High-trust contradicting: {sum(1 for s in contradicting_evidence if s.get('trust_score', 0) >= 0.9)}

TOP SUPPORTING EVIDENCE:
{json.dumps(support_summary, indent=2) if support_summary else "None"}

TOP CONTRADICTING EVIDENCE:
{json.dumps(contradict_summary, indent=2) if contradict_summary else "None"}

INSTRUCTIONS FOR GENERATING REASON:

1. Focus on SPORTS FACTS:
   - Mention specific scores, statistics, dates
   - Name the teams, players, tournaments involved
   - Reference the actual sports events or results

2. Cite AUTHORITATIVE SOURCES:
   - Mention specific sources (ESPN, NBA.com, official league sites)
   - Reference official records, box scores, or announcements
   - Use phrases like "according to ESPN" or "official NBA records show"

3. Be DIRECT and CLEAR:
   - Start with "This claim is {verdict} because..."
   - State the key facts that support the verdict
   - Be specific about what contradicts or supports the claim

4. AVOID Technical Jargon:
   - Do NOT mention "snippets", "trust scores", "weighted scores"
   - Do NOT discuss the verification process
   - Do NOT use phrases like "evidence suggests" - be direct

5. Keep it CONCISE:
   - 2-4 sentences maximum
   - Get straight to the point
   - Focus on the most important facts

GOOD EXAMPLES:

Claim: "LeBron James scored 40 points as Lakers defeated Warriors 120-110 on March 15, 2025"
Verdict: FALSE
Good Reason: "This claim is FALSE because multiple authoritative sources (ESPN, NBA.com, The Athletic) confirm that LeBron James scored 35 points, not 40 points, in the Lakers vs Warriors game on March 15, 2025. Official box scores and game reports consistently show the 35-point performance. The game score of 120-110 is accurate, but LeBron's point total is incorrect."

Claim: "Jadeja played for CSK this year but next season he will join RR"
Verdict: TRUE
Good Reason: "This claim is TRUE. According to ESPN Cricinfo and official IPL sources, Ravindra Jadeja played for Chennai Super Kings (CSK) during the 2025 IPL season. Multiple credible reports from November 2025 confirm that CSK has traded Jadeja to Rajasthan Royals (RR) ahead of the IPL 2026 auction."

Claim: "Dhoni was retired and will not play IPL anymore"
Verdict: FALSE
Good Reason: "This claim is FALSE. While MS Dhoni has not made an official retirement announcement, recent reports from ESPN Cricinfo and CSK official sources indicate he continues to be associated with the team and has not definitively ruled out playing in future IPL seasons."

BAD EXAMPLES (Avoid These):

Bad: "The claim is false because 5 high-trust snippets contradict it with a weighted score of 4.2."
Bad: "Evidence from multiple sources suggests the claim may not be accurate."
Bad: "Based on our verification process, we found contradicting information."

Now generate ONE clear, crisp reason for the claim: "{claim}"

Respond with ONLY the reason text (no JSON, no formatting, just the reason):"""
        
        try:
            # Create a temporary agent for explanation generation with date tool
            explanation_model = BedrockModel(
                model_id="amazon.nova-pro-v1:0",
                temperature=0.7,  # Slightly higher for natural language
                max_tokens=500
            )
            reason_agent = Agent(
                model=explanation_model,
                tools=[],  # No tools needed for reason generation
                system_prompt="""You are a sports fact-checker providing clear, direct verdict explanations.

CRITICAL RULES:
1. Focus on SPORTS FACTS (scores, dates, teams, players, tournaments)
2. Cite AUTHORITATIVE SOURCES by name (ESPN, NBA.com, official leagues)
3. Be DIRECT: Start with "This claim is {verdict} because..."
4. AVOID technical jargon (no "snippets", "trust scores", "verification process")
5. Keep it CONCISE (2-4 sentences)
6. State facts clearly, don't hedge with "suggests" or "may be"

Your goal: Explain WHY the verdict is what it is using sports facts and sources."""
            )
            
            result = reason_agent(reason_prompt)
            
            # Extract text from result
            if hasattr(result, 'message'):
                message = result.message
                if isinstance(message, dict):
                    if 'content' in message:
                        content = message['content']
                        if isinstance(content, list):
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict) and 'text' in item:
                                    text_parts.append(item['text'])
                                elif isinstance(item, str):
                                    text_parts.append(item)
                            reason = " ".join(text_parts)
                        else:
                            reason = str(content)
                    else:
                        reason = str(message)
                elif isinstance(message, str):
                    reason = message
                else:
                    reason = str(message)
            else:
                reason = str(result)
            
            # Clean up the reason
            reason = reason.strip()
            
            # Fallback if reason is too short or empty
            if len(reason) < 50:
                # Generate a basic reason from the verdict
                if verdict == "TRUE":
                    reason = f"This claim is TRUE. Multiple authoritative sports sources confirm the facts stated in the claim."
                elif verdict == "FALSE":
                    reason = f"This claim is FALSE. Authoritative sports sources contradict the facts stated in the claim."
                else:
                    reason = f"This claim is UNVERIFIED. Insufficient evidence from authoritative sports sources to confirm or deny the claim."
            
            return reason
            
        except Exception as e:
            print(f"  ⚠️ Could not generate reason: {e}")
            # Fallback to basic reason
            if verdict == "TRUE":
                return f"This claim is TRUE. Multiple authoritative sports sources confirm the facts stated in the claim."
            elif verdict == "FALSE":
                return f"This claim is FALSE. Authoritative sports sources contradict the facts stated in the claim."
            else:
                return f"This claim is UNVERIFIED. Insufficient evidence from authoritative sports sources to confirm or deny the claim."
    
    def _log_step(self, step: str, agent: str, input_data: Any, output_data: Any, metrics: Dict = None):
        """
        Log a workflow execution step for debugging and transparency.
        
        Each step is recorded with timestamp, agent name, inputs, and outputs.
        This creates an audit trail of the entire verification process.
        
        Args:
            step: Name of the workflow step (e.g., "classification", "decomposition")
            agent: Name of the agent executing this step
            input_data: Input data provided to the agent
            output_data: Output data produced by the agent
            metrics: Optional metrics dictionary with token usage and latency
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "agent": agent,
            "input_preview": str(input_data)[:200] if input_data else None,  # Preview for readability
            "output_preview": str(output_data)[:200] if output_data else None,
            "full_output": output_data,  # Complete output for analysis
            "metrics": metrics  # Include metrics if available
        }
        self.execution_log.append(log_entry)
    
    def _save_results(self, claim: str, results: Dict) -> str:
        """
        Save verification results to a JSON file.
        
        Creates a timestamped filename with sanitized claim text.
        Includes complete execution log and all results for reproducibility.
        
        Args:
            claim: Original claim text
            results: Complete verification results dictionary
            
        Returns:
            Path to saved file
        """
        # Create filename: YYYYMMDD_HHMMSS_sanitized_claim.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_part = sanitize_filename(claim)
        filename = f"{RESULTS_DIR}/{timestamp}_{query_part}.json"
        
        # Prepare output structure
        output = {
            "original_claim": claim,
            "timestamp": datetime.now().isoformat(),
            "workflow_version": "2.3_llm_evidence_scoring",
            "execution_log": self.execution_log,  # Complete audit trail
            "results": results  # All verification results
        }
        
        # Write to file with pretty formatting
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {filename}")
        return filename
    
    def _extract_json_from_result(self, result) -> Dict:
        """
        Extract JSON from agent result with robust parsing.
        
        LLM agents may return JSON in various formats (plain, in code blocks, etc.).
        This function tries multiple parsing strategies to extract the JSON reliably.
        
        Strategies:
        1. Direct JSON parse
        2. Brace matching to find JSON object
        3. Extract from markdown code blocks
        
        Args:
            result: Agent result object (may have various structures)
            
        Returns:
            Parsed JSON dictionary, or error dict if parsing fails
        """
        try:
            text = ""
            if hasattr(result, 'message'):
                message = result.message
                if isinstance(message, dict):
                    if 'content' in message:
                        content = message['content']
                        if isinstance(content, list):
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict) and 'text' in item:
                                    text_parts.append(item['text'])
                                elif isinstance(item, str):
                                    text_parts.append(item)
                            text = " ".join(text_parts)
                        else:
                            text = str(content)
                    else:
                        text = str(message)
                elif isinstance(message, str):
                    text = message
                else:
                    text = str(message)
            else:
                text = str(result)
            
            # Try direct JSON parse
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
            
            # Extract JSON from text using brace matching
            brace_count = 0
            start_idx = -1
            for i, char in enumerate(text):
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        try:
                            return json.loads(text[start_idx:i+1])
                        except json.JSONDecodeError:
                            pass
            
            # Try extracting from code blocks
            code_block_match = re.search(r'``````', text, re.DOTALL)
            if code_block_match:
                try:
                    return json.loads(code_block_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            print(f"⚠️ JSON extraction warning: Could not parse")
            return {"raw_output": text, "parse_error": True}
        
        except Exception as e:
            print(f"⚠️ JSON extraction error: {str(e)}")
            return {"raw_output": str(result), "error": str(e)}
    
    def _execute_parallel_searches(self, queries: List[Dict]) -> List[Dict]:
        """
        Execute multiple search queries in parallel for efficiency.
        
        Uses ThreadPoolExecutor to run searches concurrently, respecting
        MAX_PARALLEL_WORKERS limit to avoid rate limiting.
        
        Args:
            queries: List of query dictionaries with 'id', 'query', 'claim_id', etc.
            
        Returns:
            List of search result dictionaries, one per query
        """
        search_results = []
        
        def execute_single_search(query_obj: Dict) -> Dict:
            """
            Execute a single search query and return formatted results.
            
            Args:
                query_obj: Query dictionary with metadata
                
            Returns:
                Dictionary with query metadata and search results
            """
            query = query_obj.get('query', '')
            query_id = query_obj.get('id', '')
            
            try:
                print(f"  🔍 [{query_id}] {query[:70]}...")
                
                # Add delay BEFORE search to avoid rate limiting
                time.sleep(RATE_LIMIT_DELAY)
                
                # Use Perplexity Chat Completion API
                result_data = perplexity_chat_answer(query)
                
                success = result_data.get('success', False)
                # For Perplexity Chat API, we get answer and citations instead of results
                answer = result_data.get('answer', '')
                citations = result_data.get('citations', [])
                
                if success and answer:
                    print(f"  ✅ [{query_id}] Got response with {len(citations)} citations")
                else:
                    error_msg = result_data.get('error', 'No response')
                    print(f"  ❌ [{query_id}] {error_msg}")
                
                return {
                    "query_id": query_id,
                    "query": query,
                    "claim_id": query_obj.get('claim_id', 'unknown'),
                    "query_type": query_obj.get('query_type', 'unknown'),
                    "priority": query_obj.get('priority', 'medium'),
                    "confidence": query_obj.get('confidence', 0.7),
                    "rationale": query_obj.get('rationale', ''),
                    "original_claim": query_obj.get('original_claim', ''),
                    "answer": answer,
                    "citations": citations,
                    "success": success,
                    "error": result_data.get('error') if not success else None
                }
            
            except Exception as e:
                print(f"  ❌ [{query_id}] Exception: {str(e)}")
                return {
                    "query_id": query_id,
                    "query": query,
                    "claim_id": query_obj.get('claim_id', 'unknown'),
                    "query_type": query_obj.get('query_type', 'unknown'),
                    "priority": query_obj.get('priority', 'medium'),
                    "confidence": query_obj.get('confidence', 0.7),
                    "rationale": query_obj.get('rationale', ''),
                    "original_claim": query_obj.get('original_claim', ''),
                    "answer": "",
                    "citations": [],
                    "success": False,
                    "error": str(e)
                }
        
        print(f"\n  🔄 Executing {min(len(queries), NUM_SEARCH_QUERIES)} searches in parallel...")
        
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            future_to_query = {
                executor.submit(execute_single_search, q): q
                for q in queries[:NUM_SEARCH_QUERIES]
            }
            
            for future in as_completed(future_to_query):
                try:
                    result = future.result(timeout=SEARCH_TIMEOUT + 5)
                    search_results.append(result)
                except Exception as e:
                    query = future_to_query[future]
                    print(f"  ❌ [{query.get('id', '?')}] Timeout: {str(e)}")
                    search_results.append({
                        "query_id": query.get('id', 'unknown'),
                        "query": query.get('query', ''),
                        "claim_id": query.get('claim_id', 'unknown'),
                        "query_type": query.get('query_type', 'unknown'),
                        "priority": query.get('priority', 'medium'),
                        "confidence": query.get('confidence', 0.7),
                        "rationale": query.get('rationale', ''),
                        "original_claim": query.get('original_claim', ''),
                        "answer": "",
                        "citations": [],
                        "success": False,
                        "error": f"Execution timeout: {str(e)}"
                    })
        
        return search_results
    
    def classify_question_response(self, original_claim: str, question: str, perplexity_response: Dict, agent_index: int = 0) -> Dict:
        """
        Classify a Perplexity Chat Completion response using Claude 3.7 Sonnet.
        
        Workflow:
        1. Perplexity Chat API answers the verification question (already done)
        2. Claude classifies the Perplexity response as SUPPORT/CONTRADICT/IRRELEVANT
        
        Args:
            original_claim: Original claim being verified
            question: Verification question asked
            perplexity_response: Response from Perplexity Chat API
            agent_index: Index of Claude agent to use from pool
            
        Returns:
            Classification dictionary with TOON-parsed data
        """
        try:
            # Extract Perplexity answer and citations
            answer = perplexity_response.get('answer', '')
            citations = perplexity_response.get('citations', [])
            
            # Format citations for Claude
            citations_text = "\n".join([f"[{i+1}] {cite}" for i, cite in enumerate(citations)])
            
            # Create classification prompt for Claude
            classification_prompt = f"""Analyze the Perplexity response and determine if it supports, contradicts, or is irrelevant to the CLAIM.

CLAIM TO VERIFY: "{original_claim}"

VERIFICATION QUESTION: "{question}"

PERPLEXITY RESPONSE:
{answer}

CITATIONS:
{citations_text if citations_text else "No citations provided"}

Classify this response using TOON format."""
            
            # Use Claude agent from pool (round-robin)
            claude_agent = self.claude_classifier_pool[agent_index % len(self.claude_classifier_pool)]
            result = claude_agent(classification_prompt)
            
            # Extract metrics for Claude classifier
            self._extract_metrics(result, "claude_classifier")
            
            # Get the response text
            response_text = ""
            message = getattr(result, 'message', result)
            
            content = None
            if isinstance(message, dict):
                content = message.get('content')
            elif hasattr(message, 'content'):
                content = message.content
                
            if content:
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and 'text' in block:
                            response_text += block['text']
                        elif hasattr(block, 'text'):
                            response_text += block.text
                        else:
                            response_text += str(block)
                else:
                    response_text = str(content)
            else:
                response_text = str(message)
            
            # Parse TOON response
            classification = parse_toon_response(response_text)
            classification['perplexity_answer'] = answer
            classification['citations'] = citations
            classification['success'] = True
            
            return classification
            
        except Exception as e:
            return {
                "classification": "IRRELEVANT",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "key_evidence": "",
                "success": False,
                "error": str(e)
            }
    
    def verify_sub_claim(self, claim_obj: Dict, original_claim: str, classification: Dict) -> Dict:
        """
        Verify a single atomic sub-claim through Perplexity Chat Completion + Claude classification.
        
        This is the core verification function that:
        1. Uses Question Generation Agent to create verification questions
        2. Uses Perplexity Chat Completion API to answer questions with citations
        3. Uses Claude 3.7 Sonnet to classify Perplexity responses in parallel
        4. Aggregates results with counts
        
        Args:
            claim_obj: Atomic claim dictionary with 'id', 'statement', etc.
            original_claim: The original full claim for context
            classification: Classification results from Classification Agent
            
        Returns:
            Dictionary with:
            - id: Claim ID
            - statement: Claim text
            - queries: List of generated queries
            - classified_responses: All classified Perplexity responses
            - support_count: Number of supporting responses
            - contradict_count: Number of contradicting responses
            - irrelevant_count: Number of irrelevant responses
            - total_responses: Total responses collected
        """
        sub_claim_id = claim_obj.get('id')
        statement = claim_obj.get('statement')
        
        print(f"  🔄 Verifying atomic claim [{sub_claim_id}]: {statement[:50]}...")
        
        # Step 1: Generate multiple targeted questions for this sub-claim using the question generator agent
        question_prompt = f"""Generate exactly {NUM_SEARCH_QUERIES} highly targeted search queries based on the ORIGINAL CLAIM to verify the atomic claim.

ORIGINAL INPUT CLAIM (USE THIS FOR CONTEXT): "{original_claim}"

ATOMIC CLAIM TO VERIFY: "{statement}"

Context:
- Domain Classification: {classification.get('domain', 'N/A')}
- Claim Type: {classification.get('claim_type', 'N/A')}
- Urgency: {classification.get('urgency', 'N/A')}

INSTRUCTIONS:
1. Call get_current_datetime() tool FIRST to get the current date
2. Read and understand the ORIGINAL CLAIM: "{original_claim}"
3. Identify how the atomic claim relates to the original claim
4. Extract key entities, dates, numbers, and events from BOTH the original claim and atomic claim
5. Generate {NUM_SEARCH_QUERIES} diverse search queries that:
   - Use context from the ORIGINAL CLAIM
   - Specifically verify the ATOMIC CLAIM
   - Include entities and details from the original claim for better search results
6. For EACH query, provide:
   - query: The search query string (based on original claim context)
   - confidence (0.0-1.0): How confident you are this query will find relevant evidence
   - rationale: Brief explanation of how this query verifies the claim using original claim context
7. Each query should target different verification angles:
   - Direct fact verification (official records, statistics)
   - Source verification (authoritative statements, official data)
   - Expert consensus (scientific/sports authorities)
   - Statistical data (official numbers, records)
   - Contradiction checks (debunking, denials)

EXAMPLE:
If ORIGINAL CLAIM is: "LeBron James scored 40 points and Lakers won 120-110 against Warriors on March 15, 2025"
And ATOMIC CLAIM is: "LeBron James scored 40 points"
Then generate queries like:
- "LeBron James 40 points Lakers Warriors March 15 2025" (uses full context)
- "LeBron James scoring Lakers vs Warriors March 2025 official stats"
- "Lakers Warriors game March 15 2025 box score LeBron"

CRITICAL: 
- Include "original_claim": "{original_claim}" in your JSON response
- Use the ORIGINAL CLAIM context to make queries more specific and accurate
- Higher confidence (0.8-1.0) for queries that use full original claim context

Return JSON in the specified format with exactly {NUM_SEARCH_QUERIES} queries."""
        
        try:
            q_result = self.question_generator(question_prompt)
            q_data = self._extract_json_from_result(q_result)
            queries = q_data.get('queries', [])
            original_claim_from_agent = q_data.get('original_claim', original_claim)
            metrics = self._extract_metrics(q_result, "question_generator")
            
            if not queries:
                # Fallback: create a single query
                queries = [{
                    "id": f"{sub_claim_id}_q1",
                    "query": statement,
                    "claim_id": sub_claim_id,
                    "query_type": "direct_fact",
                    "priority": "high",
                    "confidence": 0.5,
                    "rationale": "Direct verification of claim",
                    "original_claim": original_claim
                }]
            else:
                # Ensure queries have proper IDs, claim_id, original_claim, and sport_context
                sport_context = f"{classification.get('sport', 'Sports')} - {classification.get('sub_category', '')} - {classification.get('context', '')}"
                for i, q in enumerate(queries, 1):
                    q['id'] = f"{sub_claim_id}_q{i}"
                    q['claim_id'] = sub_claim_id
                    q['original_claim'] = original_claim_from_agent
                    q['sport_context'] = sport_context  # Add sport context for targeted search
                    # Ensure all required fields exist with defaults
                    q.setdefault('confidence', 0.7)
                    q.setdefault('rationale', 'Verify claim using original claim context')
            
            print(f"    📝 Generated {len(queries)} search queries")
            print(f"    📋 Original Claim: {original_claim_from_agent[:80]}...")
            
            # Display query metadata summary
            avg_confidence = sum(q.get('confidence', 0.7) for q in queries) / len(queries) if queries else 0
            print(f"    🎯 Average Query Confidence: {avg_confidence:.2f}")
            
            # Show first query details as example
            if queries:
                first_q = queries[0]
                print(f"    📌 Example Query [{first_q.get('id')}]:")
                print(f"       Query: {first_q.get('query', '')[:70]}...")
                print(f"       Confidence: {first_q.get('confidence', 0.7):.2f}")
                print(f"       Rationale: {first_q.get('rationale', 'N/A')[:70]}...")
            
            if metrics:
                print(f"    📊 Tokens: {metrics['input_tokens']} in / {metrics['output_tokens']} out | Latency: {metrics['latency_ms']:.0f}ms | Cost: ${metrics['cost']:.6f}")
        except Exception as e:
            print(f"  ⚠️ Error generating questions for {sub_claim_id}: {e}")
            queries = [{
                "id": f"{sub_claim_id}_q1",
                "query": statement,
                "claim_id": sub_claim_id,
                "query_type": "direct_fact",
                "priority": "high",
                "confidence": 0.5,
                "intent": "Direct verification of claim",
                "relevance": "Directly searches for the claim statement",
                "original_claim": original_claim
            }]

        # Step 2: Get Perplexity Chat responses for all questions
        print(f"\n    📡 Asking Perplexity Chat API...")
        perplexity_responses = []
        
        for idx, query_obj in enumerate(queries):
            question = query_obj.get('query', '')
            query_id = query_obj.get('id', '')
            
            try:
                print(f"      🔍 [{query_id}] {question[:70]}...")
                
                # Add delay to avoid rate limiting
                time.sleep(RATE_LIMIT_DELAY)
                
                # Get Perplexity Chat response
                perplexity_response = perplexity_chat_answer(question)
                
                if perplexity_response.get('success'):
                    print(f"      ✅ [{query_id}] Got response with {len(perplexity_response.get('citations', []))} citations")
                    self.metrics["perplexity_chat"]["success"] += 1
                else:
                    error_msg = perplexity_response.get('error', 'No response')
                    print(f"      ❌ [{query_id}] {error_msg}")
                    self.metrics["perplexity_chat"]["failed"] += 1
                
                self.metrics["perplexity_chat"]["questions"] += 1
                
                perplexity_responses.append({
                    'query': query_obj,
                    'perplexity_response': perplexity_response
                })
                
            except Exception as e:
                print(f"      ❌ [{query_id}] Exception: {str(e)}")
                self.metrics["perplexity_chat"]["questions"] += 1
                self.metrics["perplexity_chat"]["failed"] += 1
                
                perplexity_responses.append({
                    'query': query_obj,
                    'perplexity_response': {
                        'success': False,
                        'question': question,
                        'answer': '',
                        'citations': [],
                        'error': str(e)
                    }
                })
        
        print(f"\n    ✅ Perplexity: {self.metrics['perplexity_chat']['success']} success, {self.metrics['perplexity_chat']['failed']} failed")
        
        # Step 3: Classify Perplexity responses using Claude
        print(f"\n    🤖 Classifying Perplexity responses...")
        classified_responses = []
        support_count = 0
        contradict_count = 0
        irrelevant_count = 0
        
        for idx, response_data in enumerate(perplexity_responses):
            query_obj = response_data['query']
            perplexity_response = response_data['perplexity_response']
            query_id = query_obj.get('id', '')
            question = query_obj.get('query', '')
            
            try:
                print(f"      🔍 [{query_id}] Classifying response...")
                
                # Classify the Perplexity response using Claude
                classification = self.classify_question_response(
                    original_claim=original_claim,
                    question=question,
                    perplexity_response=perplexity_response,
                    agent_index=idx
                )
                
                # Count classifications
                classification_result = classification.get('classification', 'IRRELEVANT')
                if classification_result == 'SUPPORT':
                    support_count += 1
                elif classification_result == 'CONTRADICT':
                    contradict_count += 1
                else:
                    irrelevant_count += 1
                
                classified_responses.append({
                    'query': query_obj,
                    'perplexity_response': perplexity_response,
                    'classification': classification
                })
                
                print(f"      ✅ [{query_id}] {classification_result} (confidence: {classification.get('confidence', 0.0):.2f})")
                
            except Exception as e:
                print(f"      ❌ [{query_id}] Classification error: {str(e)}")
                classified_responses.append({
                    'query': query_obj,
                    'perplexity_response': perplexity_response,
                    'classification': {
                        'classification': 'IRRELEVANT',
                        'confidence': 0.0,
                        'reasoning': f'Error: {str(e)}',
                        'success': False
                    }
                })
                irrelevant_count += 1
        
        # Summary of classified responses
        print(f"\n    ✅ Perplexity Classification Complete:")
        print(f"       Support: {support_count}")
        print(f"       Contradict: {contradict_count}")
        print(f"       Irrelevant: {irrelevant_count}")
        print(f"       Total responses: {len(classified_responses)}")
        
        return {
            "id": sub_claim_id,
            "statement": statement,
            "queries": queries,
            "classified_responses": classified_responses,
            "support_count": support_count,
            "contradict_count": contradict_count,
            "irrelevant_count": irrelevant_count,
            "total_responses": len(classified_responses)
        }
    def verify_claim(self, claim: str) -> Dict:
        """
        Main workflow orchestration for complete claim verification.
        
        This is the entry point that coordinates all agents through the full pipeline:
        
        STEP 1: Classification Agent
        - Categorizes claim by domain, type, complexity, urgency
        
        STEP 2: Decomposition Agent
        - Breaks claim into atomic sub-claims
        - Identifies dependencies between claims
        
        STEP 3: Question Generation & Parallel Verification
        - For each atomic claim:
          * Generate 10 diverse search queries
          * Execute searches in parallel
          * Collect and classify all snippets
        
        STEP 4: Final Verdict Calculation
        - Aggregate all evidence with trust scoring
        - Apply verdict rules based on evidence strength
        - Calculate confidence score
        
        Args:
            claim: The claim text to verify
            
        Returns:
            Complete results dictionary with:
            - classification: Domain, type, complexity, urgency
            - decomposition: Atomic claims and dependencies
            - sub_claim_results: Verification results for each atomic claim
            - evaluation: Final verdict, confidence, evidence summary
            - execution_log: Complete audit trail
        """
        print(f"\n{'='*80}")
        print(f"🏆 SPORTS MISINFORMATION DETECTION WORKFLOW")
        print(f"{'='*80}")
        print(f"📝 Sports Claim: {claim}")
        print(f"{'='*80}\n")
        
        # Initialize progress
        self.progress_tracker.update(5, "Initializing verification...")
        
        current_dt = datetime.now()
        date_context = f"Current Date: {current_dt.strftime('%B %d, %Y')} ({current_dt.strftime('%Y-%m-%d')})"
        
        # Step 1: Classification
        self.progress_tracker.update(10, "Classifying claim...")
        print("📋 [STEP 1/5] Classification Agent")
        print("="*80)
        classification_prompt = f"""Classify this claim:

{date_context}

Claim: "{claim}"

Provide your classification in JSON format."""
        
        try:
            classification_result = self.classifier(classification_prompt)
            classification = self._extract_json_from_result(classification_result)
            metrics = self._extract_metrics(classification_result, "classifier")
            self._log_step("classification", "classifier_agent", claim, classification, metrics)
            
            print(f"  ✅ Domain: {classification.get('domain', 'N/A')}")
            print(f"  ✅ Type: {classification.get('claim_type', 'N/A')}")
            print(f"  ✅ Complexity: {classification.get('complexity', 'N/A')}")
            if metrics:
                print(f"  📊 Tokens: {metrics['input_tokens']} in / {metrics['output_tokens']} out | Latency: {metrics['latency_ms']:.0f}ms | Cost: ${metrics['cost']:.6f}\n")
            else:
                print()
        except Exception as e:
            print(f"  ❌ Classification failed: {str(e)}")
            classification = {"error": str(e)}
        
        # Step 2: Decomposition
        self.progress_tracker.update(25, "Breaking down claim...")
        print("="*80)
        print("🧩 [STEP 2/5] Decomposition Agent")
        print("="*80)
        decomposition_prompt = f"""Break down this claim into atomic sub-claims with dependencies:

{date_context}

Claim: "{claim}"

Classification: {json.dumps(classification, indent=2)}

Provide decomposition in JSON format with dependencies identified."""
        
        try:
            decomposition_result = self.decomposer(decomposition_prompt)
            decomposition = self._extract_json_from_result(decomposition_result)
            atomic_claims = decomposition.get('atomic_claims', [])
            dependency_graph = decomposition.get('dependency_graph', {})
            metrics = self._extract_metrics(decomposition_result, "decomposer")
            self._log_step("decomposition", "decomposer_agent", classification, decomposition, metrics)
            
            print(f"  ✅ Generated {len(atomic_claims)} atomic claims")
            
            for claim_obj in atomic_claims:
                print(f"    • {claim_obj.get('id')}: {claim_obj.get('statement', '')[:60]}...")
            
            if metrics:
                print(f"  📊 Tokens: {metrics['input_tokens']} in / {metrics['output_tokens']} out | Latency: {metrics['latency_ms']:.0f}ms | Cost: ${metrics['cost']:.6f}\n")
            else:
                print()
        except Exception as e:
            print(f"  ❌ Decomposition failed: {str(e)}")
            decomposition = {"error": str(e)}
            atomic_claims = []
            dependency_graph = {}
        
        # Step 3: Parallel Verification with Question Generation and Snippet Classification
        self.progress_tracker.update(40, "Generating verification questions...")
        print("="*80)
        print("🚀 [STEP 3/4] Question Generation & Parallel Verification")
        print("="*80)
        sub_claim_results = []
        
        print(f"  🔄 Processing {len(atomic_claims)} atomic claims ({MAX_CLAIM_VERIFIERS} parallel workers)...")
        print(f"  📝 Each claim will generate {NUM_SEARCH_QUERIES} targeted search queries")
        print(f"  ⚡ Snippet classification: {MAX_SNIPPET_CLASSIFIERS} parallel workers per claim")
        print(f"  🛡️ Rate limiting: {RATE_LIMIT_DELAY}s delay to avoid API throttling\n")
        
        with ThreadPoolExecutor(max_workers=MAX_CLAIM_VERIFIERS) as executor:
            future_to_claim = {
                executor.submit(self.verify_sub_claim, claim_obj, claim, classification): claim_obj
                for claim_obj in atomic_claims
            }
            
            for future in as_completed(future_to_claim):
                try:
                    result = future.result()
                    sub_claim_results.append(result)
                except Exception as e:
                    print(f"  ❌ Sub-claim verification failed: {e}")
        
        print(f"\n  ✅ Completed verification for {len(sub_claim_results)} atomic claims\n")
        
        # Step 4: Calculate Final Verdict
        self.progress_tracker.update(85, "Calculating final verdict...")
        print("="*80)
        print("⚖️ [STEP 4/4] Calculating Final Verdict")
        print("="*80)
        
        # Aggregate all snippets with their classifications and trust scores
        all_supporting = []
        all_contradicting = []
        all_irrelevant = []
        all_snippets = []  # Track all snippets for metrics
        
        for sub_result in sub_claim_results:
            for response_data in sub_result.get('classified_responses', []):
                classification_result = response_data.get('classification', {})
                classification = classification_result.get('classification', 'IRRELEVANT')
                perplexity_response = response_data.get('perplexity_response', {})
                
                # For Perplexity responses, we don't have traditional trust scores
                # We'll use the classification confidence as a proxy
                trust_score = classification_result.get('confidence', 0.0)
                
                snippet_data = {
                    'claim_id': sub_result.get('id'),
                    'url': 'perplexity_response',  # Perplexity responses don't have individual URLs
                    'snippet': perplexity_response.get('answer', ''),
                    'trust_score': trust_score,
                    'classification_confidence': classification_result.get('confidence', 0.0),
                    'reasoning': classification_result.get('reasoning', ''),
                    'citations': perplexity_response.get('citations', [])
                }
                
                all_snippets.append(snippet_data)  # Add to all snippets list
                
                if classification == 'SUPPORT':
                    all_supporting.append(snippet_data)
                elif classification == 'CONTRADICT':
                    all_contradicting.append(snippet_data)
                else:
                    all_irrelevant.append(snippet_data)
        
        # Calculate weighted scores (trust_score * classification_confidence)
        support_score = sum(s['trust_score'] * s['classification_confidence'] for s in all_supporting)
        contradict_score = sum(s['trust_score'] * s['classification_confidence'] for s in all_contradicting)
        
        # Count high-trust sources (trust_score >= 0.9)
        high_trust_support = sum(1 for s in all_supporting if s['trust_score'] >= 0.9)
        high_trust_contradict = sum(1 for s in all_contradicting if s['trust_score'] >= 0.9)
        
        print(f"  📊 Support: {len(all_supporting)} snippets (weighted score: {support_score:.2f})")
        print(f"  📊 Contradict: {len(all_contradicting)} snippets (weighted score: {contradict_score:.2f})")
        print(f"  📊 Irrelevant: {len(all_irrelevant)} snippets")
        print(f"  🏆 High-trust support: {high_trust_support}, High-trust contradict: {high_trust_contradict}")
        
        # Determine verdict based on comprehensive rules
        verdict = "UNVERIFIED"
        confidence_score = 0.0
        reasoning = ""
        
        # Calculate total evidence
        total_relevant = len(all_supporting) + len(all_contradicting)
        total_evidence = total_relevant + len(all_irrelevant)
        
        # PRIORITY 1: Insufficient Evidence (overwhelming irrelevant data)
        # For claims like "X joined Y again", if 80%+ evidence is irrelevant (historical), claim lacks recent proof
        if total_evidence > 0 and (len(all_irrelevant) / total_evidence) > 0.8 and len(all_supporting) < 5:
            verdict = "UNVERIFIED"
            confidence_score = 20.0 + (len(all_supporting) * 5)  # Slight boost for any support found
            reasoning = f"Insufficient recent evidence ({len(all_supporting)} support, {len(all_irrelevant)} irrelevant - mostly historical/unrelated data)"
        
        # PRIORITY 2: Opinion/Derived Claims with Contradictions
        # For claims like "X is a failed captain", contradicting evidence means the claim is FALSE
        elif any(ac.get('type') == 'opinion' for ac in atomic_claims) and len(all_contradicting) > len(all_supporting):
            verdict = "FALSE"
            confidence_score = min(100, (contradict_score / (support_score + contradict_score)) * 100) if (support_score + contradict_score) > 0 else 50
            reasoning = f"Opinion claim contradicted by evidence ({len(all_contradicting)} contradict vs {len(all_supporting)} support)"
        
        # PRIORITY 3: High-Trust Source Dominance (2x ratio required)
        elif high_trust_support > 0 and high_trust_support > high_trust_contradict * 2:
            verdict = "TRUE"
            confidence_score = min(100, support_score * 10)
            reasoning = f"High-trust sources ({high_trust_support}) strongly support the claim"
        
        # Rule 2: If high-trust sources contradict significantly outweigh support, it's FALSE
        elif high_trust_contradict > 0 and high_trust_contradict > high_trust_support * 2:
            verdict = "FALSE"
            confidence_score = min(100, contradict_score * 10)
            reasoning = f"High-trust sources ({high_trust_contradict}) strongly contradict the claim"
        
        # Rule 3: If high-trust sources support and no contradictions, it's TRUE
        elif high_trust_support > 0 and len(all_contradicting) == 0:
            verdict = "TRUE"
            confidence_score = min(100, support_score * 10)
            reasoning = f"High-trust sources ({high_trust_support}) support the claim with no contradictions"
        
        # Rule 4: If support score significantly higher than contradict score
        elif support_score > contradict_score * 2 and len(all_supporting) >= 3:
            verdict = "TRUE"
            confidence_score = min(100, (support_score / (support_score + contradict_score)) * 100)
            reasoning = f"Strong support ({len(all_supporting)} snippets) outweighs contradictions"
        
        # Rule 5: If contradict score significantly higher than support score
        elif contradict_score > support_score * 2 and len(all_contradicting) >= 3:
            verdict = "FALSE"
            confidence_score = min(100, (contradict_score / (support_score + contradict_score)) * 100)
            reasoning = f"Strong contradictions ({len(all_contradicting)} snippets) outweigh support"
        
        # Rule 6: Mixed evidence - use weighted scores
        elif support_score > contradict_score:
            verdict = "TRUE"
            confidence_score = min(100, (support_score / (support_score + contradict_score)) * 100)
            reasoning = f"Evidence leans toward support ({len(all_supporting)} support vs {len(all_contradicting)} contradict)"
        
        elif contradict_score > support_score:
            verdict = "FALSE"
            confidence_score = min(100, (contradict_score / (support_score + contradict_score)) * 100)
            reasoning = f"Evidence leans toward contradiction ({len(all_contradicting)} contradict vs {len(all_supporting)} support)"
        
        # Rule 7: Insufficient or balanced evidence
        else:
            verdict = "UNVERIFIED"
            confidence_score = 50.0
            reasoning = "Insufficient or evenly balanced evidence to make a determination"
        
        # Generate clear, sports-focused reason using LLM
        self.progress_tracker.update(95, "Generating explanation...")
        print(f"\n🤖 Generating final reason...")
        reason = self._generate_final_reason(
            claim=claim,
            verdict=verdict,
            confidence_score=confidence_score,
            supporting_evidence=all_supporting[:3],  # Top 3 supporting
            contradicting_evidence=all_contradicting[:3],  # Top 3 contradicting
            atomic_claims=atomic_claims,
            sub_claim_results=sub_claim_results
        )
        
        # Calculate total metrics
        for agent_name in ["classifier", "decomposer", "question_generator", "claude_classifier"]:
            agent_metrics = self.metrics[agent_name]
            self.metrics["total"]["input_tokens"] += agent_metrics["input_tokens"]
            self.metrics["total"]["output_tokens"] += agent_metrics["output_tokens"]
            self.metrics["total"]["latency_ms"] += agent_metrics["latency_ms"]
            self.metrics["total"]["cost"] += agent_metrics["cost"]
        
        self.metrics["total"]["total_tokens"] = self.metrics["total"]["input_tokens"] + self.metrics["total"]["output_tokens"]
        
        evaluation = {
            "verdict": verdict,  # TRUE, FALSE, or UNVERIFIED
            "confidence": confidence_score,
            "reason": reason,  # Single clear reason from LLM
            "evidence_summary": {
                "total_support": len(all_supporting),
                "total_contradict": len(all_contradicting),
                "total_irrelevant": len(all_irrelevant),
                "support_weighted_score": support_score,
                "contradict_weighted_score": contradict_score,
                "high_trust_support": high_trust_support,
                "high_trust_contradict": high_trust_contradict
            },
            "top_supporting_evidence": sorted(all_supporting, key=lambda x: x['trust_score'], reverse=True)[:3],
            "top_contradicting_evidence": sorted(all_contradicting, key=lambda x: x['trust_score'], reverse=True)[:3],
            "metrics": self.metrics
        }
        
        self._log_step("evaluation", "verdict_calculator", sub_claim_results, evaluation)
        
        # Display final verdict
        print(f"\n{'='*80}")
        print(f"🎯 FINAL VERDICT")
        print(f"{'='*80}")
        
        verdict_emoji = {
            'TRUE': '✅',
            'FALSE': '❌',
            'UNVERIFIED': '⚠️'
        }
        
        emoji = verdict_emoji.get(verdict, '❓')
        print(f"{emoji} Verdict: {verdict}")
        print(f"📊 Confidence: {confidence_score:.1f}%")
        print(f"\n💬 Reason:")
        print(f"{reason}")
        
        print(f"\n📈 Evidence Summary:")
        print(f"   • Supporting sources: {len(all_supporting)}")
        print(f"   • Contradicting sources: {len(all_contradicting)}")
        print(f"   • High-trust support: {high_trust_support}")
        print(f"   • High-trust contradict: {high_trust_contradict}")
        
        print(f"\n🔍 Technical Details:")
        print(f"   • Support score: {support_score:.2f}")
        print(f"   • Contradict score: {contradict_score:.2f}")
        
        # Print detailed metrics breakdown
        print(f"\n{'='*80}")
        print(f"📊 METRICS BREAKDOWN")
        print(f"{'='*80}")
        
        print(f"\n[1] Classification Agent:")
        print(f"    Input Tokens  : {self.metrics['classifier']['input_tokens']}")
        print(f"    Output Tokens : {self.metrics['classifier']['output_tokens']}")
        print(f"    Latency (ms)  : {self.metrics['classifier']['latency_ms']:.0f}")
        print(f"    Cost          : ${self.metrics['classifier']['cost']:.6f}")
        
        print(f"\n[2] Decomposition Agent:")
        print(f"    Input Tokens  : {self.metrics['decomposer']['input_tokens']}")
        print(f"    Output Tokens : {self.metrics['decomposer']['output_tokens']}")
        print(f"    Latency (ms)  : {self.metrics['decomposer']['latency_ms']:.0f}")
        print(f"    Cost          : ${self.metrics['decomposer']['cost']:.6f}")
        
        print(f"\n[3] Question Generation Agent:")
        print(f"    Input Tokens  : {self.metrics['question_generator']['input_tokens']}")
        print(f"    Output Tokens : {self.metrics['question_generator']['output_tokens']}")
        print(f"    Latency (ms)  : {self.metrics['question_generator']['latency_ms']:.0f}")
        print(f"    Cost          : ${self.metrics['question_generator']['cost']:.6f}")
        
        print(f"\n[4] Snippet Classifier Agent ({len(all_snippets)} snippets):")
        print(f"    Input Tokens  : {self.metrics['claude_classifier']['input_tokens']}")
        print(f"    Output Tokens : {self.metrics['claude_classifier']['output_tokens']}")
        print(f"    Latency (ms)  : {self.metrics['claude_classifier']['latency_ms']:.0f}")
        print(f"    Cost          : ${self.metrics['claude_classifier']['cost']:.6f}")
        
        print(f"\n{'='*80}")
        print(f"💰 TOTAL COST & USAGE")
        print(f"{'='*80}")
        print(f"Total Input Tokens  : {self.metrics['total']['input_tokens']}")
        print(f"Total Output Tokens : {self.metrics['total']['output_tokens']}")
        print(f"Total Tokens        : {self.metrics['total']['total_tokens']}")
        print(f"Total Latency (ms)  : {self.metrics['total']['latency_ms']:.0f}")
        print(f"Total Cost          : ${self.metrics['total']['cost']:.6f}")
        
        print(f"{'='*80}\n")
        
        results = {
            "classification": classification,
            "decomposition": decomposition,
            "sub_claim_results": sub_claim_results,
            "evaluation": evaluation,
            "execution_log": self.execution_log
        }
        
        self._save_results(claim, results)
        self.progress_tracker.update(100, "Verification complete!")
        return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for interactive claim verification.
    
    Initializes the MisinformationDetector and provides an interactive
    command-line interface for verifying claims.
    
    Usage:
        python search.py
        
    Then enter claims to verify, or 'exit' to quit.
    Results are automatically saved to verification_results/ directory.
    """
    # Check for required API key
    if not PERPLEXITY_API_KEY:
        print("❌ Error: PERPLEXITY_API_KEY not found")
        print("Please create a .env file with: PERPLEXITY_API_KEY=your-key")
        return
    
    print("="*80)
    print("🚀 MISINFORMATION DETECTION SYSTEM v3.2 (SOTA Stateless Architecture)")
    print("="*80)
    print(f"\n📅 Current Date: {datetime.now().strftime('%B %d, %Y')}")
    print(f"⚙️ Configuration:")
    print(f"  - Queries per claim: {NUM_SEARCH_QUERIES}")
    print(f"  - Search workers: {MAX_PARALLEL_WORKERS} (parallel)")
    print(f"  - Snippet classifiers: {MAX_SNIPPET_CLASSIFIERS} (stateless agent pool) ⚡")
    print(f"  - Claim verifiers: {MAX_CLAIM_VERIFIERS} (parallel) ⚡")
    print(f"  - Rate limit delay: {RATE_LIMIT_DELAY}s (optimized for API limits)")
    print(f"  - Max Tokens: {MAX_TOKENS_CONFIG}")
    print(f"  - API Endpoint: https://api.perplexity.ai/search")
    print(f"  - Architecture: Stateless agents (no context accumulation)")
    print(f"  - Features: 99% cost reduction + 4-5x speedup + rate limit protection\n")
    
    try:
        detector = SportsMisinformationDetector()
        print("\n✅ System initialization complete!")
        print("="*80)
    except Exception as e:
        print(f"\n❌ Initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    while True:
        try:
            claim = input("\n🏆 Enter SPORTS claim to verify (or 'exit' to quit): ").strip()
            
            if not claim:
                continue
            
            if claim.lower() in ['exit', 'quit', 'q', 'bye']:
                print("\n👋 Goodbye!")
                break
            
            results = detector.verify_claim(claim)
            print("\n" + "="*80)
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()