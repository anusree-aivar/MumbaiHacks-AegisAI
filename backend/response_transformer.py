"""
Response Transformer Module

Transforms the new backend response format to match the current frontend expectations.
This allows the new backend to work with the existing frontend without breaking changes.
"""

from typing import Dict, Any, List
from datetime import datetime


def transform_to_legacy_format(new_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform new backend response format to legacy format expected by current frontend.
    
    New Format:
    {
        "original_claim": "...",
        "final_verdict": "TRUE|FALSE|UNVERIFIED",
        "confidence_score": 0.855,  # 0-1 range
        "explanation": "...",
        "atomic_claims": [...],
        "total_sources": 10,
        "verification_time": 45.2,
        "timestamp": "..."
    }
    
    Legacy Format:
    {
        "success": true,
        "claim": "...",
        "timestamp": "...",
        "classification": {...},
        "decomposition": {...},
        "questions": {...},
        "search_results": [...],
        "sub_claim_results": [...],
        "evaluation": {
            "overall_verdict": "VERIFIED|FALSE|UNVERIFIED",
            "confidence_score": 85.5,  # 0-100 range
            "sub_claim_verdicts": [...],
            "summary": "...",
            ...
        },
        "execution_log": [...]
    }
    
    Args:
        new_response: Response from new backend
        
    Returns:
        Transformed response in legacy format
    """
    
    # Map verdict names
    verdict_map = {
        "TRUE": "VERIFIED",
        "FALSE": "FALSE",
        "UNVERIFIED": "UNVERIFIED",
        "PARTIALLY_TRUE": "PARTIALLY_VERIFIED"
    }
    
    final_verdict = new_response.get("final_verdict", "UNVERIFIED")
    legacy_verdict = verdict_map.get(final_verdict, "UNVERIFIED")
    
    # Convert confidence from 0-1 to 0-100
    confidence_score = new_response.get("confidence_score", 0.0) * 100
    
    # Transform atomic claims to sub_claim_verdicts
    sub_claim_verdicts = []
    atomic_claims = new_response.get("atomic_claims", [])
    
    for idx, atomic_claim in enumerate(atomic_claims, 1):
        # Extract sources and convert to key_evidence format
        key_evidence = []
        for source in atomic_claim.get("sources", [])[:5]:  # Top 5 sources
            # Determine credibility tier based on trust score
            trust_score = source.get("trust_score", 0.5)
            if trust_score >= 0.9:
                credibility_tier = 1  # God Tier
            elif trust_score >= 0.7:
                credibility_tier = 2  # High Trust
            elif trust_score >= 0.5:
                credibility_tier = 3  # Medium Trust
            else:
                credibility_tier = 4  # Low Trust
            
            # Determine if it supports the claim
            classification = source.get("classification", "IRRELEVANT")
            supports_claim = classification == "SUPPORT"
            
            key_evidence.append({
                "title": source.get("title", "Unknown Source"),
                "url": source.get("url", ""),
                "credibility_tier": credibility_tier,
                "supports_claim": supports_claim
            })
        
        # Map atomic claim verdict
        atomic_verdict = atomic_claim.get("verdict", "UNVERIFIED")
        legacy_atomic_verdict = verdict_map.get(atomic_verdict, "UNVERIFIED")
        
        sub_claim_verdicts.append({
            "claim_id": f"claim_{idx}",
            "statement": atomic_claim.get("claim", ""),
            "verdict": legacy_atomic_verdict,
            "confidence": atomic_claim.get("confidence", 0.0) * 100,  # Convert to 0-100
            "supporting_count": atomic_claim.get("supporting_count", 0),
            "refuting_count": atomic_claim.get("contradicting_count", 0),
            "dependency_status": "verified",  # Simplified
            "key_evidence": key_evidence,
            "rationale": new_response.get("explanation", "")[:200]  # Truncate
        })
    
    # Create legacy response structure
    legacy_response = {
        "success": True,
        "claim": new_response.get("original_claim", ""),
        "timestamp": new_response.get("timestamp", datetime.now().isoformat()),
        
        # Classification (simplified - we don't have this in new format)
        "classification": {
            "domain": "Sports",  # Default for sports-focused system
            "claim_type": "Factual",
            "complexity": "Compound" if len(atomic_claims) > 1 else "Simple",
            "urgency": "Medium",
            "rationale": "Sports claim classification"
        },
        
        # Decomposition (reconstruct from atomic claims)
        "decomposition": {
            "original_claim": new_response.get("original_claim", ""),
            "atomic_claims": [
                {
                    "id": f"claim_{idx}",
                    "statement": ac.get("claim", ""),
                    "dependencies": [],
                    "type": "fact",
                    "entities": [],
                    "temporal": "",
                    "quantitative": "",
                    "priority": "high" if idx == 1 else "medium"
                }
                for idx, ac in enumerate(atomic_claims, 1)
            ],
            "dependency_graph": {
                "foundational": [f"claim_{idx}" for idx in range(1, len(atomic_claims) + 1)],
                "derived": []
            },
            "total_claims": len(atomic_claims)
        },
        
        # Questions (simplified - we don't track individual queries in new format)
        "questions": {
            "current_date_used": datetime.now().strftime("%Y-%m-%d"),
            "queries": [],
            "total_queries": new_response.get("total_sources", 0),
            "strategy_rationale": "Generated targeted search queries for verification"
        },
        
        # Search results (empty - new format doesn't expose raw search results)
        "search_results": [],
        
        # Sub-claim results (map from atomic_claims)
        "sub_claim_results": [
            {
                "id": f"claim_{idx}",
                "statement": ac.get("claim", ""),
                "queries": [],
                "snippets": [
                    {
                        "url": s.get("url", ""),
                        "snippet": s.get("snippet", ""),
                        "trust_score": s.get("trust_score", 0.5),
                        "classification": s.get("classification", "IRRELEVANT"),
                        "domain": s.get("url", "").split("/")[2] if "/" in s.get("url", "") else "unknown"
                    }
                    for s in ac.get("sources", [])
                ],
                "support_count": ac.get("supporting_count", 0),
                "contradict_count": ac.get("contradicting_count", 0),
                "irrelevant_count": len(ac.get("sources", [])) - ac.get("supporting_count", 0) - ac.get("contradicting_count", 0),
                "total_snippets": len(ac.get("sources", []))
            }
            for idx, ac in enumerate(atomic_claims, 1)
        ],
        
        # Evaluation (main transformation)
        "evaluation": {
            "overall_verdict": legacy_verdict,
            "confidence_score": confidence_score,
            "sub_claim_verdicts": sub_claim_verdicts,
            "dependency_analysis": {
                "foundational_claims_verified": final_verdict == "TRUE",
                "broken_dependencies": [],
                "notes": "All claims verified independently"
            },
            "summary": new_response.get("explanation", ""),
            "key_findings": [
                f"Verdict: {legacy_verdict}",
                f"Confidence: {confidence_score:.1f}%",
                f"Sources analyzed: {new_response.get('total_sources', 0)}",
                f"Verification time: {new_response.get('verification_time', 0):.1f}s"
            ],
            "limitations": "Verification based on available online sources at the time of analysis."
        },
        
        # Execution log (empty - new format doesn't expose detailed logs to API)
        "execution_log": [
            {
                "timestamp": new_response.get("timestamp", datetime.now().isoformat()),
                "step": "verification_complete",
                "agent": "multi_agent_system",
                "output_preview": f"Verdict: {legacy_verdict}, Confidence: {confidence_score:.1f}%"
            }
        ],
        
        "error": None
    }
    
    return legacy_response


def transform_news_response(new_news_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform new news API response to legacy format.
    
    New Format:
    {
        "articles": [
            {
                "id": "1",
                "title": "...",
                "description": "...",
                "url": "...",
                "source": "...",
                "publishedAt": "...",
                "imageUrl": "...",
                "verified": null
            }
        ],
        "total": 10,
        "timestamp": "..."
    }
    
    Legacy Format:
    {
        "success": true,
        "articles": [
            {
                "id": "1",
                "title": "...",
                "summary": "...",  # description -> summary
                "source": "...",
                "imageUrl": "...",
                "verificationStatus": "unverified",  # verified -> verificationStatus
                "timestamp": "...",  # publishedAt -> timestamp
                "category": "Sports",
                "url": "..."
            }
        ],
        "total": 10,
        "error": null
    }
    
    Args:
        new_news_response: Response from new news API
        
    Returns:
        Transformed response in legacy format
    """
    
    # Map verified status
    verified_map = {
        "TRUE": "verified",
        "FALSE": "false",
        "UNVERIFIED": "unverified",
        "PARTIALLY_TRUE": "unverified",
        None: "unverified"
    }
    
    legacy_articles = []
    for article in new_news_response.get("articles", []):
        verified_status = article.get("verified")
        legacy_status = verified_map.get(verified_status, "unverified")
        
        legacy_articles.append({
            "id": article.get("id", ""),
            "title": article.get("title", ""),
            "summary": article.get("description", ""),  # description -> summary
            "source": article.get("source", ""),
            "imageUrl": article.get("imageUrl"),
            "verificationStatus": legacy_status,
            "timestamp": article.get("publishedAt", ""),  # publishedAt -> timestamp
            "category": "Sports",  # Default category
            "url": article.get("url", "")
        })
    
    return {
        "success": True,
        "articles": legacy_articles,
        "total": new_news_response.get("total", len(legacy_articles)),
        "error": None
    }
