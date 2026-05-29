"""Task-framing system instruction.

Slim by design: the response schema (schema.py) enforces every attribute's values and format, so this only frames the task.
Passed as `system_instruction` so it stays a static prefix (images are the variable, last-position content).
"""

PROMPT = """You annotate frames sampled from a single gait-recognition video: the same person appears in every frame, captured by a fixed surveillance-style camera. Describe the person and scene for each frame, staying consistent across frames of the same sequence. Return exactly one caption per frame, in the same order the frames are given."""
