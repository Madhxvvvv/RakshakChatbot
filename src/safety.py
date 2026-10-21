"""Safety guardrails for the Healthcare AI Chatbot.

Implements pre-generation input checking (emergency detection, diagnosis/prescription
prevention) and post-generation output validation. These are deterministic rules,
so identical queries always route identically and escalation works even if the
LLM API is down.
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
    # Warning signs of severe dengue / malaria / typhoid / TB, when described as
    # happening to someone ("I am", "my son has", "she keeps") rather than asked about.
    r"\b(?:i|i'm|im|i've|my\s+\w+|he|she|they|he's|she's|they're)\b[^.?!]{0,40}"
    r"(?:vomiting\s+blood|(?:coughing|cough(?:s|ed)?)\s+(?:up\s+)?blood"
    r"|blood\s+in\s+(?:\w+\s+)?(?:vomit|stool|stools|urine|poo)"
    r"|black\s+(?:tarry\s+)?stools?|bleeding\s+(?:from\s+(?:the\s+)?)?(?:gums|nose)"
    r"|severe\s+(?:abdominal|stomach|belly)\s+pain|(?:keeps?|persistent(?:ly)?)\s+vomiting"
    r"|can(?:not|'t|t)\s+(?:keep\s+(?:any\s+)?(?:fluids|water|anything)\s+down|drink)"
    r"|convulsions?|fits\b|very\s+(?:drowsy|confused)|confused\s+and|dark\s+urine"
    r"|cold\s+(?:and\s+)?clammy)",
]

# =============================================================================
# Diagnosis Request Patterns
# =============================================================================

# Regex alternation for the five in-scope diseases, used in the patterns below
_DISEASE = (
    r"(?:dengu\w*|mal[ae]?ria\w*|typh?(?:oi|io)d\w*|tyfoid\w*|enteric\s+fever"
    r"|tuberc?u?losis|t\.?b\b|chi[ck]+[ue]n?g[ou]n[iy]*[ae]?\w*)"
)

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
    # Personal diagnosis of the five in-scope diseases
    rf"\b(?:do|could|might|can)\s+(?:i|we)\s+(?:have|be\s+having|have\s+got|got)\s+{_DISEASE}",
    rf"\b(?:does|could|might|can)\s+(?:my\s+\w+|he|she|they|it)\s+(?:have|has|be|got)\s+{_DISEASE}",
    rf"\b(?:is|could|might|can)\s+(?:it|this|that|mine)\s+(?:be\s+)?{_DISEASE}",
    rf"\b(?:am\s+i|is\s+(?:my\s+\w+|he|she))\s+(?:infected\s+with|suffering\s+from)\s+{_DISEASE}",
    # Interpreting a personal test report
    r"\b(?:my|his|her|our)\s+(?:\w+\s+){0,3}(?:report|test|result|results|count|smear)\b"
    r"[^?]*\b(?:mean|normal|positive|negative|serious|dangerous|okay|ok)\b",
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
    # Treatment choice for the five in-scope diseases
    rf"\b(?:which|what|best|good)\s+(?:\w+\s+){{0,2}}(?:medicine|medication|drug|tablet|antibiotic|injection)s?\b[^?]*\b{_DISEASE}",
    # Taking or giving a named medicine (stopping is excluded: "can I stop TB
    # medicines early?" is a misconception the knowledge base answers)
    r"\b(?:can|should|may|could)\s+(?:i|we|he|she|they|my\s+\w+)\s+(?:take|give|use|start|have)\s+"
    r"(?:\w+\s+){0,3}(?:paracetamol|acetaminophen|ibuprofen|aspirin|crocin|dolo|calpol|combiflam"
    r"|antibiotics?|chloroquine|artemisinin|azithromycin|ciprofloxacin|syrup|injection|drip)",
    r"\b(?:dose|dosage)\s+(?:of|for)\b",
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

        # Scope (the five diseases only) is decided after retrieval by the
        # similarity floor in RAGPipeline, not here.
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

        # Normalise citation brackets: some models emit 【2】 instead of [2]
        response = re.sub(r"【\s*(\d+)\s*】", r"[\1]", response)

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
            for marker in [
                "disclaimer",
                "not a substitute",
                "consult",
                "healthcare provider",
                "medical advice",
            ]
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

