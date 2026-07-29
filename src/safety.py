"""Safety guardrails for the Healthcare AI Chatbot.

Implements pre-generation input checking (emergency detection, diagnosis/prescription
prevention, off-topic filtering) and post-generation output validation.
All safety checks run BEFORE the LLM is called to avoid unnecessary API usage
and ensure instant responses for critical situations.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SafetyAction(Enum):
    """Actions the safety layer can take."""

    PROCEED = "proceed"  # Safe to send to LLM
    EMERGENCY = "emergency"  # Bypass LLM, return emergency response
    BLOCK_DIAGNOSIS = "block_diagnosis"  # Redirect: diagnosis request
    BLOCK_PRESCRIPTION = "block_prescription"  # Redirect: prescription request
    REDIRECT_OFF_TOPIC = "redirect_off_topic"  # Redirect: non-health query


@dataclass
class SafetyResult:
    """Result of a safety check on user input.

    Attributes:
        action: The recommended action to take.
        reason: Human-readable explanation of why this action was chosen.
        matched_terms: Any specific terms/patterns that triggered the action.
    """

    action: SafetyAction
    reason: str
    matched_terms: Optional[list[str]] = None


# =============================================================================
# Emergency Keywords & Patterns
# =============================================================================

# Terms that strongly indicate a medical emergency
_EMERGENCY_KEYWORDS = [
    "chest pain",
    "heart attack",
    "stroke",
    "can't breathe",
    "cannot breathe",
    "cant breathe",
    "difficulty breathing",
    "not breathing",
    "stopped breathing",
    "choking",
    "severe bleeding",
    "heavy bleeding",
    "won't stop bleeding",
    "unconscious",
    "passed out",
    "fainted",
    "unresponsive",
    "having a seizure",
    "having seizures",
    "having convulsions",
    "overdose",
    "poisoning",
    "swallowed poison",
    "suicidal",
    "suicide",
    "want to die",
    "kill myself",
    "end my life",
    "self harm",
    "self-harm",
    "anaphylaxis",
    "allergic reaction severe",
    "severe allergic reaction",
    "drowning",
    "electrocution",
    "severe burn",
    "third degree burn",
    "head injury",
    "head trauma",
    "spinal injury",
    "broken neck",
    "stabbed",
    "gunshot",
    "shot wound",
]

# Regex patterns for more flexible emergency detection
_EMERGENCY_PATTERNS = [
    r"can\s*(?:not|'t|t)\s*breathe",
    r"(?:i|he|she|they)\s+(?:am|is|are)\s+(?:having\s+a\s+)?(?:heart\s+attack|stroke)",
    r"(?:want|going)\s+to\s+(?:die|kill|end|hurt)\s+(?:my|him|her|them)self",
    r"(?:someone|he|she)\s+(?:is|has)\s+(?:not\s+)?(?:breathing|conscious|responsive)",
    r"(?:severe|heavy|uncontrollable)\s+(?:pain|bleeding)",
    r"(?:i|he|she|they)\s+(?:overdosed|took\s+too\s+(?:many|much))",
]

# =============================================================================
# Diagnosis Request Patterns
# =============================================================================

_DIAGNOSIS_KEYWORDS = [
    "what disease do i have",
    "what's wrong with me",
    "what is wrong with me",
    "diagnose me",
    "diagnose my",
    "what condition do i have",
    "do i have cancer",
    "do i have diabetes",
    "what illness do i have",
    "what am i suffering from",
    "tell me my diagnosis",
    "give me a diagnosis",
    "what's my diagnosis",
    "am i sick with",
]

_DIAGNOSIS_PATTERNS = [
    r"(?:what|which)\s+(?:disease|illness|condition|disorder|syndrome)\s+do\s+i\s+have",
    r"(?:can\s+you|please)\s+diagnose",
    r"do\s+i\s+(?:have|suffer\s+from)\s+(?:a\s+)?(?:\w+\s+){0,3}(?:disease|cancer|disorder|syndrome|illness)",
    r"(?:tell|give)\s+me\s+(?:my|a)\s+diagnosis",
    r"(?:i\s+think\s+i\s+have|could\s+i\s+have).*(?:confirm|verify|check)",
]

# =============================================================================
# Prescription Request Patterns
# =============================================================================

_PRESCRIPTION_KEYWORDS = [
    "what medicine should i take",
    "what medication should i take",
    "prescribe me",
    "prescribe something",
    "what drug should i",
    "what pills should i",
    "recommend a medicine",
    "recommend medication",
    "what should i take for",
    "give me a prescription",
    "write me a prescription",
    "what dosage should i",
    "how much medicine should i",
]

_PRESCRIPTION_PATTERNS = [
    r"(?:what|which)\s+(?:medicine|medication|drug|pill|tablet)\s+(?:should|can|do)\s+i\s+(?:take|use)",
    r"(?:can\s+you|please)\s+(?:prescribe|recommend)\s+(?:a\s+)?(?:medicine|medication|drug)",
    r"(?:give|write)\s+(?:me\s+)?(?:a\s+)?prescription",
    r"(?:what|how\s+much)\s+(?:dosage|dose)\s+(?:should|do)\s+i",
    r"(?:should|can)\s+i\s+(?:take|use|try)\s+(?:\w+\s+){0,2}(?:mg|tablet|pill|capsule)",
]

# =============================================================================
# Off-Topic Detection
# =============================================================================

# Health-related keywords that indicate the query IS on-topic
_HEALTH_KEYWORDS = [
    "health", "medical", "medicine", "doctor", "hospital", "symptom", "disease",
    "illness", "pain", "ache", "fever", "cold", "flu", "cough", "headache",
    "nutrition", "diet", "exercise", "vitamin", "mineral", "protein",
    "calories", "weight", "bmi", "blood", "pressure", "sugar", "diabetes",
    "heart", "lung", "liver", "kidney", "brain", "bone", "muscle", "joint",
    "skin", "rash", "allergy", "allergic", "infection", "virus", "bacteria",
    "vaccine", "vaccination", "immunization", "antibiotic", "therapy",
    "mental health", "anxiety", "depression", "stress", "sleep", "insomnia",
    "pregnancy", "prenatal", "postnatal", "baby", "child", "pediatric",
    "first aid", "cpr", "bandage", "wound", "burn", "fracture", "sprain",
    "cancer", "tumor", "cholesterol", "thyroid", "asthma", "arthritis",
    "hydration", "water", "wellness", "fitness", "yoga", "meditation",
    "dental", "teeth", "eye", "vision", "ear", "hearing", "stomach",
    "digestion", "nausea", "vomiting", "diarrhea", "constipation",
    "breathing", "respiratory", "immune", "inflammation", "swelling",
    "fatigue", "tired", "dizzy", "dizziness", "faint", "chest",
    "body", "physical", "preventive", "screening", "checkup",
    "healthy", "unhealthy", "recover", "treatment", "heal", "cure",
    "surgical", "surgery", "operation", "therapy", "rehabilitation",
    "pharmacy", "pharmaceutical", "supplement", "herbal", "remedy",
    "covid", "coronavirus", "pandemic", "hygiene", "sanitation",
    "calorie", "carb", "fat", "fiber", "food", "eating", "meal",
]


class SafetyGuardrails:
    """Pre-generation and post-generation safety checks for the healthcare chatbot.

    Usage:
        guardrails = SafetyGuardrails()
        result = guardrails.check_input("I'm having chest pain")
        if result.action == SafetyAction.EMERGENCY:
            return EMERGENCY_RESPONSE
    """

    def check_input(self, query: str) -> SafetyResult:
        """Run all safety checks on user input.

        Checks are ordered by severity — emergency is checked first.

        Args:
            query: The raw user input string.

        Returns:
            SafetyResult with the recommended action.
        """
        if not query or not query.strip():
            return SafetyResult(
                action=SafetyAction.PROCEED,
                reason="Empty query — will be handled by UI validation.",
            )

        query_lower = query.lower().strip()

        # 1. Emergency detection (highest priority)
        emergency_result = self._detect_emergency(query_lower)
        if emergency_result:
            return emergency_result

        # 2. Diagnosis request detection
        diagnosis_result = self._detect_diagnosis_request(query_lower)
        if diagnosis_result:
            return diagnosis_result

        # 3. Prescription request detection
        prescription_result = self._detect_prescription_request(query_lower)
        if prescription_result:
            return prescription_result

        # 4. Off-topic handling — delegated to the LLM
        # Instead of fragile keyword matching, the LLM's system prompt instructs
        # it to stay on health topics and politely redirect non-health queries.
        # This is more accurate because the LLM understands context, synonyms,
        # meta-conversational queries, and nuance that keyword lists cannot.

        # All safety checks passed — let the LLM handle it
        return SafetyResult(
            action=SafetyAction.PROCEED,
            reason="Input passed all safety checks.",
        )

    def validate_output(self, response: str) -> str:
        """Post-generation validation of LLM output.

        Checks the generated response for any content that may have
        slipped through the system prompt constraints.

        Args:
            response: The LLM-generated response text.

        Returns:
            The validated (and potentially modified) response.
        """
        from src.prompts import DISCLAIMER_TEXT

        # Check if the response contains potentially unsafe prescriptive language
        prescriptive_patterns = [
            r"(?:you\s+should|i\s+recommend)\s+(?:take|taking)\s+\d+\s*(?:mg|ml|tablet|pill|capsule)",
            r"(?:take|use)\s+\d+\s*(?:mg|ml)\s+(?:of\s+)?\w+\s+(?:daily|twice|three\s+times)",
        ]

        for pattern in prescriptive_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                response += (
                    "\n\n⚠️ **Important:** Any mentions of specific dosages above are for "
                    "general educational context only. Always consult your doctor or pharmacist "
                    "for proper dosing recommendations specific to your situation."
                )
                break

        # Ensure disclaimer is present for health-related responses
        if not any(
            marker in response.lower()
            for marker in ["disclaimer", "not a substitute", "consult", "healthcare provider"]
        ):
            response += f"\n\n{DISCLAIMER_TEXT}"

        return response

    # -------------------------------------------------------------------------
    # Private detection methods
    # -------------------------------------------------------------------------

    def _detect_emergency(self, query_lower: str) -> Optional[SafetyResult]:
        """Check for emergency-related keywords and patterns."""
        matched = []

        # Keyword matching
        for keyword in _EMERGENCY_KEYWORDS:
            if keyword in query_lower:
                matched.append(keyword)

        # Regex pattern matching
        for pattern in _EMERGENCY_PATTERNS:
            if re.search(pattern, query_lower):
                match = re.search(pattern, query_lower)
                if match:
                    matched.append(match.group())

        if matched:
            return SafetyResult(
                action=SafetyAction.EMERGENCY,
                reason="Emergency-related content detected. Bypassing LLM.",
                matched_terms=list(set(matched)),
            )
        return None

    def _detect_diagnosis_request(self, query_lower: str) -> Optional[SafetyResult]:
        """Check for explicit diagnosis requests."""
        matched = []

        for keyword in _DIAGNOSIS_KEYWORDS:
            if keyword in query_lower:
                matched.append(keyword)

        for pattern in _DIAGNOSIS_PATTERNS:
            if re.search(pattern, query_lower):
                match = re.search(pattern, query_lower)
                if match:
                    matched.append(match.group())

        if matched:
            return SafetyResult(
                action=SafetyAction.BLOCK_DIAGNOSIS,
                reason="Diagnosis request detected.",
                matched_terms=list(set(matched)),
            )
        return None

    def _detect_prescription_request(self, query_lower: str) -> Optional[SafetyResult]:
        """Check for explicit prescription/medication requests."""
        matched = []

        for keyword in _PRESCRIPTION_KEYWORDS:
            if keyword in query_lower:
                matched.append(keyword)

        for pattern in _PRESCRIPTION_PATTERNS:
            if re.search(pattern, query_lower):
                match = re.search(pattern, query_lower)
                if match:
                    matched.append(match.group())

        if matched:
            return SafetyResult(
                action=SafetyAction.BLOCK_PRESCRIPTION,
                reason="Prescription/medication request detected.",
                matched_terms=list(set(matched)),
            )
        return None

    def _detect_off_topic(self, query_lower: str) -> Optional[SafetyResult]:
        """Check if the query is clearly unrelated to healthcare.

        Uses a generous allowlist approach — only redirects when there's
        high confidence the query is off-topic. Borderline queries are
        allowed through so the LLM (with its system prompt constraints)
        can handle them more nuancedly.
        """
        # Short queries (under 3 words) get the benefit of the doubt
        words = query_lower.split()
        if len(words) < 3:
            return None

        # Check for simple greetings/acknowledgments that should always pass
        greeting_starters = [
            "hi", "hello", "hey", "thanks", "thank you", "okay", "ok",
            "yes", "no", "good morning", "good evening", "good night",
        ]
        for starter in greeting_starters:
            if query_lower.startswith(starter):
                return None

        # Check for meta-conversational keywords (queries about the chat itself)
        # These should always pass through so the LLM can use conversation history
        meta_keywords = [
            "question", "query", "asked", "said", "last", "previous", "earlier",
            "first", "second", "third", "above", "before", "conversation",
            "chat", "history", "remember", "mentioned", "told", "summarize",
            "summary", "repeat", "again", "clarify", "explain more", "what did",
            "you said", "i said", "go back", "follow up", "more about",
            "tell me more", "answer", "response", "replied", "recall", "recap",
        ]
        for keyword in meta_keywords:
            if keyword in query_lower:
                return None

        # Check if any health keyword appears in the query
        for keyword in _HEALTH_KEYWORDS:
            if keyword in query_lower:
                return None

        # If we get here, the query likely has nothing to do with health
        return SafetyResult(
            action=SafetyAction.REDIRECT_OFF_TOPIC,
            reason="Query does not appear to be health-related.",
        )
