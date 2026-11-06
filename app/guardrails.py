import logging
import re
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

log = logging.getLogger("guardrails")

MAX_INPUT_CHARS = 2000

PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "US_SSN",
    "AADHAAR_NUMBER",
    "PAN_NUMBER",
]

INJECTION_PATTERNS = {
    "ignore_instructions": re.compile(
        r"\b(ignore|disregard|forget)\b\s+(all\s+|any\s+|the\s+|your\s+|my\s+)*"
        r"(previous|prior|above|earlier|preceding|original|system)\s+"
        r"(instructions?|prompts?|rules?|messages?|context)",
        re.I,
    ),
    "ignore_everything": re.compile(
        r"\b(ignore|disregard|forget)\s+(everything|all)\s+(above|before|prior|previously|you were told)",
        re.I,
    ),
    "reveal_prompt": re.compile(
        r"\b(reveal|show|print|repeat|display|leak)\b.{0,30}\b(system|hidden|initial|original)\b.{0,15}\b(prompt|instructions?|message)",
        re.I | re.S,
    ),
    "role_override": re.compile(
        r"\b(you are now|from now on you|act as|pretend (to be|you are))\b.{0,60}"
        r"\b(unrestricted|unfiltered|jailbroken|no restrictions|without (any )?(rules|restrictions|filters))",
        re.I | re.S,
    ),
    "jailbreak_terms": re.compile(r"\b(jailbreak|developer mode|dan mode|do anything now)\b", re.I),
    "fake_delimiters": re.compile(
        r"(<\s*/?\s*(system|assistant|instructions?)\s*>|\[/?INST\]|###\s*(system|instruction)s?\b)",
        re.I,
    ),
    "new_instructions": re.compile(r"\b(new|updated|additional) instructions?\s*:", re.I),
}


class GuardrailViolation(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@lru_cache
def get_analyzer() -> AnalyzerEngine:
    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    ).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="AADHAAR_NUMBER",
            patterns=[Pattern("aadhaar", r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b", 0.5)],
            context=["aadhaar", "aadhar", "uid"],
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="PAN_NUMBER",
            patterns=[Pattern("pan", r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", 0.85)],
            context=["pan", "permanent account number"],
        )
    )
    return analyzer


@lru_cache
def get_anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()


def redact(text: str) -> str:
    """Replace personal data with placeholders such as <EMAIL_ADDRESS>."""
    if not text:
        return text
    results = get_analyzer().analyze(
        text=text, language="en", entities=PII_ENTITIES, score_threshold=0.4
    )
    if not results:
        return text
    log.info("pii redacted: %s", sorted({r.entity_type for r in results}))
    return get_anonymizer().anonymize(text=text, analyzer_results=results).text


def scan_injection(text: str) -> list[str]:
    return [name for name, pattern in INJECTION_PATTERNS.items() if pattern.search(text)]


def sanitize_input(text: str, user_id: int) -> str:
    """Front door: block attacks and too-long input, then mask personal data."""
    if len(text) > MAX_INPUT_CHARS:
        raise GuardrailViolation(
            "too_long", f"Input is longer than {MAX_INPUT_CHARS} characters"
        )
    hits = scan_injection(text)
    if hits:
        log.warning("guardrail blocked user=%s reason=prompt_injection patterns=%s", user_id, hits)
        raise GuardrailViolation(
            "prompt_injection",
            "Request blocked: it looks like an attempt to override the assistant's instructions",
        )
    return redact(text)


def filter_injected(chunks: list[dict]) -> list[dict]:
    """Documents: keep chunks that contain instructions aimed at the AI out of results."""
    kept = []
    for chunk in chunks:
        hits = scan_injection(chunk["content"])
        if hits:
            log.warning(
                "guardrail dropped chunk=%s document=%s patterns=%s",
                chunk["chunk_id"], chunk["document_id"], hits,
            )
        else:
            kept.append(chunk)
    return kept

def strip_placeholders(text: str) -> str:
    """Remove masking labels like <EMAIL_ADDRESS> so they don't dilute the search."""
    cleaned = re.sub(r"<[A-Z_]+>", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip() or text    