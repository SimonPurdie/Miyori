import math
from datetime import datetime

class ImportanceScorer:
    """Calculates importance scores and handles time-based decay."""

    @staticmethod
    def calculate_importance(user_msg: str, miyori_msg: str) -> float:
        """
        Calculate importance score.
        Returns 0-1 float.
        """
        score = 0.5  # baseline
        
        # Simple heuristics
        # Explicit request
        if "remember" in user_msg.lower():
            score += 0.3
            
        # Personal keywords
        personal_keywords = ["i am", "i want", "i like", "my name", "i feel", "i work"]
        if any(kw in user_msg.lower() for kw in personal_keywords):
            score += 0.2
            
        # Decision/Goal
        if "i will" in user_msg.lower() or "promise" in user_msg.lower():
            score += 0.25
            
        # Caps at 1.0
        return min(score, 1.0)

    @staticmethod
    def get_decayed_score(base_score: float, timestamp_iso: str) -> float:
        """
        Calculate decayed importance score based on age.
        """
        try:
            timestamp = datetime.fromisoformat(timestamp_iso)
            age_days = (datetime.now() - timestamp).days
            if age_days <= 0:
                return base_score
                
            # High importance items decay MUCH slower (longer half-life)
            half_life = 100 * base_score 
            if half_life <= 0: return 0.0
            
            decay = math.exp(-age_days * math.log(2) / half_life)
            return base_score * decay
        except Exception:
            return base_score
