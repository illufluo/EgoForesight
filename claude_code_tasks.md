# Task 1: Improve V2 Prompt (do this first)

## Context
V2 pipeline is working but the prompt needs improvement. See `output_standard.md` in project knowledge for the full explanation/prediction quality standard. V1 is left as-is for now (control group).

## What to Change

### v2/prompt.py
Rewrite `build_prompt()` to produce better explanation and prediction outputs. Key requirements from the standard:

**Explanation must:**
- Describe action progression across frames ("first... then... finally..."), not a single summary
- Name specific objects ("white cup", "knife"), not vague words ("items", "objects")
- Include hand details (which hand does what)
- Start with action verbs, NOT "The person is currently..."
- No meta-descriptions ("as seen in the frames")
- 30-50 words

**Prediction must:**
- Give one clear, specific next action prediction
- Same style, format, and length as explanation (30-50 words)
- Start with action verbs, use dynamic verbs
- Brief reasoning linking to current action trend
- No vague hedging ("possibly", "might")

**Important prompt design notes:**
- Do NOT hardcode specific seconds or intervals in the prompt (e.g., don't say "0.5s interval" or "2s window"). The prompt should work with any frame interval. Say something like "consecutive frames at regular intervals" instead.
- Tell the model the frames are in chronological order
- Give a concrete example of good output in the prompt (few-shot) to stabilize output style
- Keep the Explanation:/Prediction: format labels for parsing

### v2/run.py
- Check that `_parse_response()` handles edge cases (e.g., when VLM doesn't follow format). Currently if parsing fails, both explanation and prediction get the full raw_response — this is acceptable as fallback, but add a `.strip()` to remove leading `\n` from parsed fields.

## Test
After changes, run on the test video and compare output quality with previous results.

---