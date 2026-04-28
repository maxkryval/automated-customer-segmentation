
import json


def build_llm_prompt(campaign_prompt: str, segment_summary: dict) -> str:
    return f"""
You are a strict data-driven marketing analyst.

Your task is to evaluate whether this customer segment is relevant for the campaign.

Campaign description:
{campaign_prompt}

Segment data:
{json.dumps(segment_summary, ensure_ascii=False, indent=2)}

Rules:
- Use only the provided segment data.
- Do not invent external behavior, strategy, channels, creative ideas, or campaign settings.
- Focus only on whether this segment should be targeted.
- Evaluate relevance based on numeric differences and SHAP feature contributions.
- Be conservative. If the segment does not clearly match the campaign, mark it as do_not_target.
- The priority score must be from 0 to 100:
  - 80-100 = strong target
  - 50-79 = possible secondary target
  - 0-49 = not recommended
- Explicitly list the top SHAP-valued features, using up to 3 features from top_shap_features.
- Use SHAP-valued features as the main evidence for why the segment is or is not relevant.
- Briefly describe the segment using numeric metric differences.
- Use demographic_summary when available, especially age and sex, but do not overstate demographic differences.

Return valid JSON only with this structure:

{{
  "cluster": <number>,
  "user_share_percent": <number>,
  "priority_score": <number from 0 to 100>,
  "targeting_decision": "target / secondary / do_not_target",
  "top_shap_features": [
    {{
      "feature": "...",
      "mean_abs_shap_value": <number>
    }}
  ],
  "metric_description": "...",
  "demographic_description": "...",
  "reasoning": "...",
  "final_recommendation": "..."
}}
"""


def generate_segment_recommendations(
    campaign_prompt: str,
    segment_summaries: list[dict],
    openai_api_key: str,
    openai_model: str,
    temperature: float = 0.1,
) -> list[dict]:
    if not openai_api_key:
        return []

    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)
    outputs = []

    for summary in segment_summaries:
        prompt = build_llm_prompt(campaign_prompt, summary)
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict data-driven marketing analyst. "
                        "Return valid JSON only. "
                        "Do not invent campaign strategy. "
                        "Only evaluate segment relevance based on provided data."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        content = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"OpenAI returned invalid JSON:\n{content}") from exc
        parsed["source"] = "openai"
        outputs.append(parsed)

    return outputs
