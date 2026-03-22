# Output Standard: Explanation & Prediction

> This document defines the format and quality standards for explanation and prediction outputs.
> Used by: V1-V5 inference prompts, data annotation pipeline, evaluation.

## General Principles

- Explanation and prediction follow the **same format and style**, because:
  - prediction(t) should be directly comparable to explanation(t+1)
  - In V3, both are compressed to keywords for history input — uniform format simplifies this
- Frame interval and window size are **configurable parameters**, not hardcoded
- Prediction scope: "the next action" — not tied to a specific number of seconds. Let the action's natural rhythm determine the scope.

## Explanation

**Purpose**: Describe what the person is doing in the current time window, capturing the progression of action across frames.

### Must Include
- **Action sequence**: Describe how the action changes over time ("first... then... finally..."), not a single summary sentence
- **Specific objects**: Use concrete names ("white cup", "knife", "wooden board"), avoid vague terms ("items", "objects", "things")
- **Hand details**: Which hand does what ("right hand grasps the cup")

### Must NOT Include
- Meta-descriptions about the frames ("as seen in the sequence", "across the 4 frames")
- Future predictions (that belongs to Prediction)
- Irrelevant scene descriptions (room layout, lighting — unless directly related to the action)

### Style
- Length: **30–50 words**
- Start with action, NOT "The person is currently..."
- Use dynamic verbs (reaches, grasps, lifts, places, stirs), avoid static descriptions (is holding, is near, is positioned)

### Example
> "The right hand picks up a thin stirrer and stirs inside the right cup on the round wooden board. Then the hand releases the stirrer and grasps the cup, beginning to lift it from the surface."

## Prediction

**Purpose**: Predict the most likely next action the person will perform, based on the current action's trajectory.

### Must Include
- **One clear action prediction** — a specific, concrete next action
- **Brief reasoning** — one short phrase linking it to the current action trend

### Must NOT Include
- Meta-descriptions
- Multiple alternative predictions (at most two if genuinely equally likely)
- Vague hedging ("possibly", "might", "could potentially")

### Style
- Length: **30–50 words** (same as explanation)
- Start with action, same dynamic verb style as explanation
- Same level of object and hand detail as explanation

### Example
> "The person lifts the white cup toward their mouth with the right hand and takes a sip, then begins to lower the cup back toward the round wooden board with a relaxed grip."

## V1 Note

V1 uses a single frame, so:
- Explanation: Describes the static scene and the apparent action at that instant (no temporal progression). Still follows the same style rules.
- Prediction: Same standard as V2/V3.
