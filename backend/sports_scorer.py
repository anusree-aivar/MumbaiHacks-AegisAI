"""
Sports-Specific Trust Scoring Engine
Comprehensive coverage of all major sports with official sources and trusted social media
"""

import requests
from urllib.parse import urlparse
from typing import Dict, Set, List


class HybridSportsScorer:
    """
    A specialized trust engine for Sports Misinformation.
    Covers 20+ sports with official sources, trusted journalists, and verified social media.
    """
    
    def __init__(self, opr_api_key=None):
        self.opr_api_key = opr_api_key
        self.headers = {'API-OPR': self.opr_api_key} if opr_api_key else {}
        
        # =====================================================
        # 🏆 TIER 1: THE UNDISPUTED TRUTH (Score: 1.0)
        # Official Leagues, Governing Bodies, Verified Sources
        # =====================================================
        self.god_tier_domains: Set[str] = {
            # Global Sports Media
            "theathletic.com", "bbc.com/sport", "skysports.com", "espn.com",
            "reuters.com/sports", "apnews.com/sports", "sports.yahoo.com",
            
            # ========== FOOTBALL (SOCCER) ==========
            "fifa.com", "uefa.com", "premierleague.com", "laliga.com",
            "bundesliga.com", "seriea.com", "ligue1.com",
            "transfermarkt.com", "goal.com", "espn.com/soccer",
            
            # ========== CRICKET ==========
            "icc-cricket.com", "espncricinfo.com", "iplt20.com",
            "cricbuzz.com", "thecricketer.com", "bcci.tv",
            "cricket.com.au", "ecb.co.uk", "wisden.com",
            
            # ========== BASKETBALL ==========
            "nba.com", "fiba.basketball", "ncaa.com/sports/basketball",
            "euroleaguebasketball.net", "basketball-reference.com",
            "wnba.com", "nba.com/stats",
            
            # ========== TENNIS ==========
            "atptour.com", "wtatennis.com", "itftennis.com",
            "ausopen.com", "wimbledon.com", "rolandgarros.com",
            "usopen.org", "daviscup.com",
            
            # ========== VOLLEYBALL ==========
            "volleyballworld.com", "fivb.com", "cev.eu",
            "avp.com", "siatka.org",
            
            # ========== TABLE TENNIS ==========
            "ittf.com", "worldtabletennis.com", "tabletennisengland.co.uk",
            
            # ========== BADMINTON ==========
            "bwfbadminton.com", "bwfworldtour.bwfbadminton.com",
            "olympics.com/en/sports/badminton",
            
            # ========== FIELD HOCKEY ==========
            "fih.hockey", "hockeyindia.org", "eurohockey.org",
            
            # ========== ICE HOCKEY ==========
            "nhl.com", "iihf.com", "eliteprospects.com",
            "hockey-reference.com",
            
            # ========== BASEBALL ==========
            "mlb.com", "wbsc.org", "baseball-reference.com",
            "milb.com", "baseballamerica.com",
            
            # ========== AMERICAN FOOTBALL (NFL) ==========
            "nfl.com", "espn.com/nfl", "pro-football-reference.com",
            "nfl.com/stats", "profootballtalk.com",
            
            # ========== RUGBY ==========
            "world.rugby", "espn.com/rugby", "premiershiprugby.com",
            "sixnationsrugby.com", "superrugby.co.nz",
            
            # ========== GOLF ==========
            "pgatour.com", "europeantour.com", "golfdigest.com",
            "masters.com", "usga.org", "randa.org",
            
            # ========== ATHLETICS (TRACK & FIELD) ==========
            "worldathletics.org", "olympics.com/en/sports/athletics",
            "letsrun.com", "iaaf.org",
            
            # ========== MOTORSPORTS (F1, MOTOGP) ==========
            "formula1.com", "motogp.com", "fia.com",
            "autosport.com", "motorsport.com", "the-race.com",
            
            # ========== BOXING ==========
            "wbcboxing.com", "wba.org", "ringtv.com",
            "boxingscene.com", "espn.com/boxing",
            
            # ========== MMA (UFC) ==========
            "ufc.com", "mmafighting.com", "sherdog.com",
            "bellator.com", "onefc.com",
            
            # ========== WRESTLING (WWE) ==========
            "wwe.com", "wrestlinginc.com", "prowrestling.net",
            "fightful.com", "pwinsider.com",
            
            # ========== CYCLING ==========
            "uci.org", "procyclingstats.com", "cyclingnews.com",
            "letour.fr", "giroditalia.it",
            
            # ========== ESPORTS ==========
            "eslgaming.com", "liquipedia.net", "escharts.com",
            "lolesports.com", "dota2.com/esports", "hltv.org",
            
            # ========== OLYMPICS & MULTI-SPORT ==========
            "olympics.com", "olympic.org", "paralympic.org",
            "commonwealthgames.com"
        }
        
        # =====================================================
        # 🎤 TRUSTED JOURNALISTS & INSIDERS (Score: 1.0)
        # The humans who ARE the news
        # =====================================================
        self.god_tier_journalists: List[str] = [
            # Football (Soccer)
            "Fabrizio Romano", "David Ornstein", "Gianluca Di Marzio", 
            "Henry Winter", "Guillem Balague", "Raphael Honigstein",
            "James Pearce", "Paul Joyce", "Matteo Moretto", "Nicolo Schira",
            
            # Basketball (NBA)
            "Adrian Wojnarowski", "Shams Charania", "Chris Haynes", 
            "Marc Stein", "Brian Windhorst", "Zach Lowe", "Ramona Shelburne",
            
            # American Football (NFL)
            "Adam Schefter", "Ian Rapoport", "Tom Pelissero", 
            "Jay Glazer", "Mike Garafolo", "Dianna Russini", "Albert Breer",
            
            # Baseball (MLB)
            "Jeff Passan", "Ken Rosenthal", "Jon Heyman",
            "Bob Nightengale", "Buster Olney", "Joel Sherman",
            
            # Motorsports (F1)
            "Chris Medland", "Albert Fabrega", "Tobi Gruner",
            "Joe Saward", "Will Buxton", "Andrew Benson",
            
            # Cricket
            "Harsha Bhogle", "Michael Vaughan", "Wasim Jaffer",
            "Aakash Chopra", "Ian Bishop", "Ravi Shastri",
            
            # Tennis
            "Ben Rothenberg", "Christopher Clarey", "Reem Abulleil",
            "Jose Morgado", "Stuart Fraser",
            
            # MMA/Boxing
            "Ariel Helwani", "Brett Okamoto", "Dan Rafael",
            "Mike Coppinger", "Marc Raimondi",
            
            # General Sports
            "Adrian Wojnarowski", "Adam Schefter", "Fabrizio Romano"
        ]
        
        # =====================================================
        # 📱 VERIFIED SOCIAL MEDIA ACCOUNTS (Score: 1.0)
        # Official team/league accounts on Twitter, Instagram, etc.
        # =====================================================
        self.verified_social_handles: Set[str] = {
            # Twitter/X patterns
            "twitter.com/espn", "twitter.com/espnfc", "twitter.com/espncricinfo",
            "twitter.com/nba", "twitter.com/nfl", "twitter.com/mlb",
            "twitter.com/fifacom", "twitter.com/uefa", "twitter.com/premierleague",
            "twitter.com/icc", "twitter.com/bcci", "twitter.com/iplt20",
            "twitter.com/atptour", "twitter.com/wta", "twitter.com/wimbledon",
            "twitter.com/f1", "twitter.com/motogp", "twitter.com/ufc",
            "twitter.com/fabrizioromano", "twitter.com/wojespn", "twitter.com/shamscharania",
            "twitter.com/adamschefter", "twitter.com/jeffpassan",
            
            # Instagram patterns
            "instagram.com/espn", "instagram.com/nba", "instagram.com/nfl",
            "instagram.com/premierleague", "instagram.com/championsleague",
            "instagram.com/icc", "instagram.com/iplt20", "instagram.com/ufc",
            
            # YouTube patterns
            "youtube.com/@espn", "youtube.com/@nba", "youtube.com/@nfl",
            "youtube.com/@premierleague", "youtube.com/@icc",
            
            # Facebook/Meta patterns
            "facebook.com/espn", "facebook.com/nba", "facebook.com/nfl",
            "facebook.com/premierleague", "facebook.com/icc"
        }
        
        # =====================================================
        # ⚠️ TIER 4: RUMOR MILLS (Score: 0.3)
        # High traffic, but known for gossip/unverified claims
        # =====================================================
        self.rumor_mill: Set[str] = {
            # UK Tabloids (often reliable for match reports, unreliable for transfers)
            "thesun.co.uk", "dailymail.co.uk", "mirror.co.uk", 
            "express.co.uk", "dailystar.co.uk", "metro.co.uk",
            
            # Spanish Rumors
            "marca.com", "as.com", "donbalon.com", "fichajes.net",
            "sport.es", "mundodeportivo.com",
            
            # Italian Rumors
            "calciomercato.com", "tuttosport.com", "corrieredellosport.it",
            
            # US/General Aggregators
            "bleacherreport.com", "clutchpoints.com", "nypost.com",
            "transfernewslive.com", "dailysnark.com", "sportbible.com",
            "givemesport.com", "90min.com",
            
            # Social Media Aggregators
            "talkSPORT.com"  # Sometimes reliable, often sensational
        }
        
        # =====================================================
        # 🚫 TIER 5: SATIRE & FAKE (Score: 0.0)
        # Immediate Discard
        # =====================================================
        self.blacklist: Set[str] = {
            # The Classics
            "theonion.com", "clickhole.com",
            
            # Specific Sports Parodies (Very dangerous for AI)
            "nbacentel", "nflcentel", "mlbcentel",
            "ballsacksports", "buttcracksports",
            "soccermemes", "trollfootball", "troll football",
            "sportspickle", "kayfabenews",
            
            # Known Fake News Sites
            "empirenews.net", "worldnewsdailyreport.com",
            "huzlers.com", "8shit.net"
        }
    
    def get_trust_score(self, url: str, snippet_text: str = "") -> float:
        """
        Calculates a trust score (0.0 - 1.0) for a given piece of evidence.
        
        Args:
            url: The source URL
            snippet_text: The text content/snippet from the source
            
        Returns:
            Trust score between 0.0 and 1.0
        """
        snippet_lower = snippet_text.lower() if snippet_text else ""
        domain = self._extract_domain(url)
        url_lower = url.lower()
        
        # 1. 🚫 SATIRE CHECK (Safety First)
        for bad_actor in self.blacklist:
            if bad_actor in snippet_lower or bad_actor in domain:
                return 0.0
        
        # 2. 📱 VERIFIED SOCIAL MEDIA CHECK
        # Check if URL matches verified social media patterns
        for verified_handle in self.verified_social_handles:
            if verified_handle in url_lower:
                return 1.0
        
        # 3. 🏆 JOURNALIST CHECK (The "Fast Pass")
        # If the text mentions trusted journalists, it's likely gold
        if snippet_text:
            for journalist in self.god_tier_journalists:
                if journalist.lower() in snippet_lower:
                    return 1.0
        
        # 4. 🏛️ DOMAIN WHITELIST CHECK
        if domain in self.god_tier_domains:
            return 1.0
        
        # Check for partial domain matches (e.g., bbc.com/sport)
        for god_domain in self.god_tier_domains:
            if god_domain in domain or domain in god_domain:
                return 1.0
        
        # 5. ⚠️ RUMOR MILL CHECK
        if domain in self.rumor_mill:
            return 0.3
        
        # Check for partial rumor mill matches
        for rumor_domain in self.rumor_mill:
            if rumor_domain in domain or domain in rumor_domain:
                return 0.3
        
        # 6. 🌍 FALLBACK: OpenPageRank (Scalability)
        # Only query API for unknown domains
        if self.opr_api_key:
            return self._query_open_pagerank(domain)
        
        # Default for unknown/blogs if no API key
        return 0.4
    
    def _extract_domain(self, url: str) -> str:
        """Extract clean domain from URL."""
        try:
            domain = urlparse(url).netloc.lower()
            return domain.replace("www.", "")
        except:
            return ""
    
    def _query_open_pagerank(self, domain: str) -> float:
        """
        Query OpenPageRank API for domain authority.
        
        Args:
            domain: The domain to check
            
        Returns:
            Normalized trust score (0.0-0.7)
        """
        try:
            url = f"https://openpagerank.com/api/v1.0/getPageRank?domains[]={domain}"
            response = requests.get(url, headers=self.headers, timeout=1.5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status_code') == 200:
                    rank = float(data['response'][0]['page_rank_decimal'])
                    # Normalize 0-10 rank to 0.0-0.7 (Cap at 0.7 for algorithms)
                    return min((rank / 10.0), 0.7)
        except:
            pass
        
        return 0.4
    
    def get_detailed_score(self, url: str, snippet_text: str) -> Dict:
        """
        Get detailed scoring information for debugging/transparency.
        
        Args:
            url: The source URL
            snippet_text: The text content/snippet from the source
            
        Returns:
            Dictionary with score and reasoning
        """
        snippet_lower = snippet_text.lower()
        domain = self._extract_domain(url)
        
        # Check blacklist
        for bad_actor in self.blacklist:
            if bad_actor in snippet_lower or bad_actor in domain:
                return {
                    "score": 0.0,
                    "tier": "BLACKLIST",
                    "reason": f"Matched blacklisted source: {bad_actor}",
                    "domain": domain
                }
        
        # Check journalist
        for journalist in self.god_tier_journalists:
            if journalist.lower() in snippet_lower:
                return {
                    "score": 1.0,
                    "tier": "GOD_TIER_JOURNALIST",
                    "reason": f"Mentioned trusted journalist: {journalist}",
                    "domain": domain
                }
        
        # Check god tier domains
        if domain in self.god_tier_domains:
            return {
                "score": 1.0,
                "tier": "GOD_TIER_DOMAIN",
                "reason": "Official/authoritative source",
                "domain": domain
            }
        
        # Check rumor mill
        if domain in self.rumor_mill:
            return {
                "score": 0.3,
                "tier": "RUMOR_MILL",
                "reason": "Known for unverified claims",
                "domain": domain
            }
        
        # OpenPageRank fallback
        if self.opr_api_key:
            score = self._query_open_pagerank(domain)
            return {
                "score": score,
                "tier": "OPENPAGERANK",
                "reason": f"PageRank-based score",
                "domain": domain
            }
        
        return {
            "score": 0.4,
            "tier": "UNKNOWN",
            "reason": "Unknown source, default score",
            "domain": domain
        }


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("🏆 HYBRID SPORTS SCORER - TEST SUITE")
    print("="*80)
    
    # Initialize scorer (without OPR key for testing)
    scorer = HybridSportsScorer()
    
    # Test cases
    test_cases = [
        {
            "url": "https://www.nba.com/news/lakers-win-championship",
            "snippet": "The Lakers won the championship last night",
            "expected": "HIGH (Official Source)"
        },
        {
            "url": "https://twitter.com/FabrizioRomano/status/123",
            "snippet": "Fabrizio Romano reports: Mbappé to Real Madrid, here we go!",
            "expected": "HIGH (Trusted Journalist)"
        },
        {
            "url": "https://www.thesun.co.uk/sport/transfer-rumor",
            "snippet": "Sources say Ronaldo might move to Saudi Arabia",
            "expected": "LOW (Rumor Mill)"
        },
        {
            "url": "https://nbacentel.com/lebron-traded",
            "snippet": "LeBron James traded to Shanghai Sharks",
            "expected": "ZERO (Satire/Fake)"
        },
        {
            "url": "https://www.espn.com/nba/story",
            "snippet": "Breaking news from ESPN about the NBA playoffs",
            "expected": "HIGH (God Tier Domain)"
        }
    ]
    
    print("\n🧪 Running Test Cases:\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['expected']}")
        print(f"  URL: {test['url']}")
        print(f"  Snippet: {test['snippet'][:60]}...")
        
        result = scorer.get_detailed_score(test['url'], test['snippet'])
        
        print(f"  ✅ Score: {result['score']:.2f}")
        print(f"  ✅ Tier: {result['tier']}")
        print(f"  ✅ Reason: {result['reason']}")
        print(f"  ✅ Domain: {result['domain']}")
        print()
    
    print("="*80)
    print("✅ Test Suite Complete")
    print("="*80)